# DK 混合模型设计文档

## 1. 概述

### 1.1 什么是 DK 模型

DK（DeepSeek-Kimi）是一个混合模型，在 DeepSeek V4 主干网络中以固定间隔插入 Kimi KDA（Kolmogorov-Arnold Delta Attention）线性注意力层，形成”密集注意力 + 线性注意力”的混合架构。

具体地，DK 在 DeepSeek V4 的第 11、21、31 层（1-indexed）替换为标准 Kimi KDA 层，其余层保持 DeepSeek V4 的 MLA（Multi-Head Latent Attention）+ MoE FFN 结构不变。

### 1.2 源模型规格

DK 模型由两个预训练 checkpoint 组装而成：

**DeepSeek V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash`)：

| 参数 | 值 |
|------|-----|
| 层数 | 43 |
| hidden_size | 4096 |
| hc_mult | 4 |
| 注意力类型 | MLA + Indexer（稀疏注意力 top-k） |
| 路由专家 | 256（FP4）+ 1 共享专家（BF16） |
| 每 token 激活专家数 | 6 |
| 注意力模式 | 2 全注意力 + 21 压缩(4×) + 20 滑窗(128) |
| KV cache (FP8) | 1024 bytes/token/layer（K_latent 512 + V_latent 512） |
| vocab_size | 129,280 |

**Kimi Linear 48B-A3B** (`moonshotai/Kimi-Linear-48B-A3B-Instruct`)：

| 参数 | 值 |
|------|-----|
| 层数 | 27 |
| hidden_size | 2304 |
| KDA 层数 | 20（位置 [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26]） |
| 全注意力层数 | 7（位置 [4,8,12,16,20,24,27]） |
| 路由专家 | 256（BF16）+ 1 共享专家 |
| 每 token 激活专家数 | 8 |
| KDA head_dim | 128 |
| KDA num_heads | 32 |
| short_conv_kernel_size | 4 |
| vocab_size | 163,840 |

### 1.3 设计原则

- **单文件模型**：模型作为单文件放在 `vllm/model_executor/models/` 下，复用 `WeightsMapper` + `AutoWeightsLoader` 机制做权重映射。
- **最大化复用**：DK 的 DeepSeek V4 层直接复用现有 `DeepseekV4DecoderLayer`，Kimi KDA 层复用 `KimiDecoderLayer`，仅新增投影层和分发逻辑。
- **与 DeepSeek V4 兼容**：DK 模型在非 KDA 位置的行为与 DeepSeek V4 完全一致，所有 MHC、MLA、MoE 逻辑不变。

---

## 2. 架构设计

### 2.1 整体结构

```
DKForCausalLM (顶层)
├── DKModel (内部模型)
│   ├── VocabParallelEmbedding (embed_tokens)
│   ├── DKDecoderLayer × N (层分发器)
│   │   ├── [KDA 位置] → KimiKDALayer
│   │   │   ├── ProjectionIn  (MHC multi-stream → flat)
│   │   │   ├── KimiDecoderLayer (KDA attention + FFN)
│   │   │   └── ProjectionOut (flat → MHC multi-stream)
│   │   └── [非 KDA 位置] → DeepseekV4DecoderLayer
│   │       ├── MLA Attention + Compressor + Indexer
│   │       ├── MHC Pre/Post (Multi-Head Collapse)
│   │       └── MoE FFN
│   ├── RMSNorm (final norm)
│   └── hc_head (MHC collapse → single stream)
├── ParallelLMHead (lm_head)
└── LogitsProcessor
```

### 2.2 数据流

```
input_ids [B, L]
    │
    ▼
Embedding + MHC expansion
    │ (B, L, hc_mult, hidden_size)
    ▼
┌──────────────────────────────────────────┐
│  For layer in 0..N-1:                    │
│    if layer_idx+1 in kda_layers:         │
│      KimiKDALayer (线性注意力)            │
│    else:                                 │
│      DeepseekV4DecoderLayer (MLA + MoE)  │
└──────────────────────────────────────────┘
    │ (B, L, hc_mult, hidden_size)
    ▼
hc_head collapse → (B, L, hidden_size)
    │
    ▼
RMSNorm → lm_head → logits
```

### 2.3 MHC（Multi-Head Collapse）机制

MHC 是 DeepSeek V4 的核心创新之一。输入经过 Embedding 后扩展为 `hc_mult` 个并行的残差流（multi-stream），每一层通过 MHC pre/post 操作在流之间进行信息混合：

- **MHC Pre**：对 multi-stream 残差做 RMSNorm → 线性变换 → Sigmoid 门控 + Sinkhorn 归一化的组合矩阵 → 加权求和 → 输出单流 `layer_input`
- **MHC Post**：将层的输出与原始 multi-stream 残差通过组合矩阵和门控重新混合 → 输出新的 multi-stream 残差

MHC 计算依赖两个关键 kernel：
1. **DeepGEMM `tf32_hc_prenorm_gemm`**：split-K 矩阵乘法（F32 精度）
2. **TileLang `mhc_pre_big_fuse_tilelang` / `mhc_post_tilelang`**：融合了 norm、门控、Sinkhorn 迭代、残差混合的 GPU kernel

这两个 kernel 仅在数据中心 GPU（H100/B200/GB300）上可用，消费级 GPU（如 RTX 5090）由于 DeepGEMM 不支持其 compute capability 而无法运行。

---

## 3. 关键设计决策

### 3.1 KimiKDALayer 的投影机制

KDA 层面临一个维度不匹配问题：

| 组件 | 输入形状 | 输出形状 |
|------|---------|---------|
| DeepSeek V4 残差流 | `(tokens, hc_mult, dv_hidden)` | 同上 |
| Kimi KDA 层 | `(tokens, kimi_hidden)` | 同上 |

解决方案：为每个 KDA 层添加可学习的线性投影层 `ProjectionIn` / `ProjectionOut`。

```python
ProjectionIn:  (tokens, hc_mult * dv_hidden) → (tokens, kimi_hidden) + RMSNorm
ProjectionOut: (tokens, kimi_hidden) → (tokens, hc_mult * dv_hidden) + RMSNorm
```

投影权重在权重组装时随机初始化（`N(0, 0.02)`），训练后通过 fine-tuning 学习。

### 3.2 KDA 层索引映射

DK 和 Kimi 使用不同的 KDA 层位置编号：

- DK：`kda_layers = [11, 21, 31]`（1-indexed，在 40 层 DeepSeek V4 中的绝对位置）
- Kimi：`linear_attn_config.kda_layers = [5, 15, 25]`（1-indexed，在 32 层 Kimi 中的位置）

之所以需要映射，是因为权重组装时从两个**预训练 checkpoint** 拼凑——Kimi checkpoint 第 5 层的权重需要加载到 DK 的第 11 层。而 `KimiDecoderLayer` 内部根据 `layer_idx` 查询 `kimi_config.linear_attn_config.kda_layers` / `full_attn_layers` 来判断自己是 KDA 还是 full attention 层，因此必须传入 Kimi 原始层号。

`KimiKDALayer.__init__` 中建立两者的映射：

```python
dk_pos = layer_idx + 1                                          # DK 的 1-indexed 位置
kimi_kda_pos = kda_layers_kimi[kda_layers_dk.index(dk_pos)]    # 对应 Kimi 的 1-indexed 位置
kimi_layer_idx = kimi_kda_pos - 1                               # Kimi 的 0-indexed 层号
```

如果是**从头训练 DK 模型**而非拼凑预训练权重，可以将两个配置直接对齐（`kda_layers` 和 `kimi_config.linear_attn_config.kda_layers` 使用相同的层号列表），此时映射退化为恒等映射 `kimi_kda_pos == dk_pos`，无需额外处理。

### 3.3 DKDecoderLayer 的分发模式

`DKDecoderLayer` 是一个轻量分发器，根据当前层是否在 `kda_layers` 中决定使用哪种层实现：

```python
class DKDecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, ...):
        layer_idx = extract_layer_index(prefix)
        if (layer_idx + 1) in config.kda_layers:
            self.layer = KimiKDALayer(...)
        else:
            self.layer = DeepseekV4DecoderLayer(...)

    def forward(self, x, positions, input_ids):
        return self.layer(x, positions, input_ids)
```

注意 `self.layer` 的命名约定使得模块路径中多一层 `.layer.`，这影响了所有权重键名的映射（见 4.2 节）。

### 3.4 混合模型接口

`DKForCausalLM` 实现四个关键接口以确保 vLLM 引擎正确管理 KV cache 和 mamba state：

| 接口 | 用途 |
|------|------|
| `HasInnerState` | 标记模型需要 mamba state 管理 |
| `IsHybrid` | 标记模型混合使用 attention 和 linear attention |
| `SupportsPP` | 支持流水线并行 |
| `MixtureOfExperts` | 支持 MoE 专家并行 |

引擎根据这些标记：
- 为 DeepSeek V4 层分配标准 KV cache
- 为 KDA 层分配 mamba state（conv state + delta state）
- 正确处理 prefill/decode 阶段的 state 传递

---

## 4. 文件结构与职责

### 4.1 核心文件

```
vllm/
├── model_executor/
│   ├── models/
│   │   ├── dk.py                    ← DK 模型主体（DKDecoderLayer, DKModel, DKForCausalLM）
│   │   ├── config.py                ← VerifyAndUpdateConfig 注册（可选）
│   │   └── deepseek_v4.py           ← DeepSeek V4 基础（复用）
│   └── layers/
│       ├── mhc.py                   ← MHC pre/post kernel 封装
│       └── activation.py            ← SwiGLU clamp 简化（删除 alpha/beta 参数）
├── transformers_utils/
│   └── configs/
│       └── dk.py                    ← DKConfig（继承 DeepseekV4Config）
├── models/
│   └── dk/
│       ├── weight_loader.py         ← 权重组装工具（合并 V4 + Kimi 权重）
│       └── deploy_gb300.sh          ← GB300 部署脚本
scripts/
└── generate_dk_test_weights.py      ← 测试用 checkpoint 生成器
tests/
└── models/
    └── test_dk.py                   ← 单元测试 & 端到端推理测试
```

### 4.2 权重命名与映射

DK 使用 HF 原始命名约定（通过 `WeightsMapper` 映射到 vLLM 内部参数路径）：

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "layers.": "model.layers.",    # 前缀映射
        "embed.": "model.embed.",
        "norm.": "model.norm.",
        "hc_head": "model.hc_head",
    },
    orig_to_new_suffix={
        "head.weight": "lm_head.weight",     # 后缀映射
        "embed.weight": "embed_tokens.weight",
    },
    orig_to_new_substr={
        ".attn.compressor.": ".attn.mla_attn.compressor.",  # 子串替换
    },
)
```

关键映射示例：

| HF 原始名 | vLLM 内部名 |
|-----------|------------|
| `embed.weight` | `model.embed_tokens.weight` |
| `head.weight` | `lm_head.weight` |
| `model.layers.0.layer.hc_attn_fn` | `model.layers.0.layer.hc_attn_fn` |
| `model.layers.1.layer.proj_in.proj.weight` | `model.layers.1.layer.proj_in.proj.weight` |

> **注意**：模块路径中的 `.layer.` 来自 `DKDecoderLayer` 的 `self.layer = ...` 命名，权重生成时必须在 HF 名称中包含此前缀。

---

## 5. 配置系统

### 5.1 DKConfig

```python
class DKConfig(DeepseekV4Config):
    model_type = "dk"

    def __init__(self, kda_layers=None, kimi_config=None, **kwargs):
        self.kda_layers = kda_layers or [11, 21, 31]
        self.kimi_config = kimi_config or {}
        super().__init__(**kwargs)
```

继承 `DeepseekV4Config`，额外包含：
- `kda_layers`：KDA 层的 1-indexed 位置列表
- `kimi_config`：Kimi 线性模型的完整 HF config 字典（传递给 `KimiLinearConfig` 解析）

### 5.2 模型注册

在 `vllm/transformers_utils/config.py` 的 `_CONFIG_REGISTRY` 中添加 `dk="DKConfig"`，使 HF transformers 能识别 `model_type: "dk"` 的 checkpoint。

---

## 6. 权重组装流程

`vllm/models/dk/weight_loader.py` 提供了一个独立的 checkpoint 组装工具，将 DeepSeek V4 和 Kimi 的权重合并为 DK checkpoint：

1. 加载 DeepSeek V4-Flash 全部权重作为基础
2. 对每个 `kda_layers` 中的层：
   - 删除该层的 DeepSeek V4 权重
   - 从 Kimi-Linear-48B 中提取对应的 KDA 层权重，重命名后插入
3. 为每个 KDA 层初始化 `ProjectionIn` / `ProjectionOut` 权重（`N(0, 0.02)`）
4. 生成 `kimi_config`，填充 DeepSeek V4 字段，标注 `model_type: "dk"`
5. 保存为 safetensors + config.json

---

## 7. GB300 部署配置

### 7.1 模型规格

DK 模型由 DeepSeek V4 Flash 和 Kimi Linear 48B-A3B 的 KDA 层组装而成：

| 组件 | 来源 | 层数 | 说明 |
|------|------|------|------|
| V4 层 (非 KDA) | DeepSeek V4 Flash | 40 | MLA + MoE，包含全注意力/压缩/滑窗三种模式 |
| KDA 层 | Kimi Linear 48B | 3 | 位置 11, 21, 31，线性注意力替代标准注意力 |

**V4 层注意力分布**（基于 `compress_ratios`，移除 KDA 位置后）：

| 注意力类型 | 层数 | `compress_ratio` | KV cache 特征 |
|-----------|------|------------------|---------------|
| 全注意力 | 2 | 0 | 存储完整上下文 KV |
| 压缩注意力 | 21 | 4 | KV 压缩 4× |
| 滑窗注意力 | 17 | 128 | KV 限定最近 128 token |
| KDA (线性注意力) | 3 | — | 无 KV cache，使用 recurrent state |

### 7.2 权重内存预算

**DeepSeek V4 部分（40 层）：**

| 组件 | 精度 | 单层参数量 | 40 层总计 |
|------|------|-----------|----------|
| MLA 注意力 + Indexer + Compressor | BF16 | ~151M | ~12.1 GB |
| MHC pre/post | FP32 | ~0.8M | ~0.13 GB |
| MoE 路由门控 | BF16 | ~1M | ~0.08 GB |
| 共享专家 | BF16 | ~25M | ~2.0 GB |
| 路由专家 (×256) | FP4 | ~25M/专家 = 6.4B | ~128.8 GB |
| Token Embedding | BF16 | ~530M | ~1.06 GB |
| LM Head | BF16 | ~530M | ~1.06 GB |
| **V4 合计** | | | **~145.2 GB** |

**Kimi KDA 部分（3 层）：**

| 组件 | 精度 | 单层参数量 | 3 层总计 |
|------|------|-----------|----------|
| KDA 注意力 (Q/K/V/O/gate/conv) | BF16 | ~47M | ~0.28 GB |
| ProjectionIn + ProjectionOut | BF16 | ~75M | ~0.45 GB |
| MoE 路由门控 | BF16 | ~0.6M | ~0.004 GB |
| 共享专家 | BF16 | ~7M | ~0.04 GB |
| 路由专家 (×256) | FP4 | ~1.8B | ~2.73 GB |
| **KDA 合计** | | | **~3.5 GB** |

**全模型总计：**

| 类别 | 大小 |
|------|------|
| BF16 参数（密集 + 注意力 + 嵌入） | ~17.3 GB |
| FP4 参数（路由专家） | ~131.5 GB |
| **模型权重总计** | **~148.8 GB** |

### 7.3 KV Cache + Mamba State 内存预算

本节以一条 2M token 请求为例，逐项计算 V4 层 KV cache 和 KDA 层 mamba state 的内存占用。

#### 7.3.1 V4 层单 token KV cache 大小

DeepSeek V4 的 MLA 使用 `fp8_ds_mla` 格式存储 KV cache（`kv_cache_interface.py:335`）：

| KV 组件 | 维度 | 精度 | 大小 |
|---------|------|------|------|
| K_nope (NoPE 部分) | `head_dim - qk_rope_head_dim = 512 - 64 = 448` | FP8 | 448 B |
| K_rope (RoPE 部分) | `qk_rope_head_dim = 64` | FP16 | 128 B |
| FP8 scale | 1 | FP8 | 8 B |
| **单层单 token 合计** | | | **584 B** |

> K_nope 是 MLA 压缩后的潜在表示（`kv_a_proj_with_mqa` 将 hidden 4096 压缩到 512 dim，其中 nope 部分 448 dim）。K_rope 必须保留 FP16 精度以正确应用 RoPE。V 不单独存储——注意力计算时通过 `kv_b_proj` 从 K_latent 重建。

#### 7.3.2 compress_ratio 对 KV 存储量的影响

`compress_ratios` **直接影响 KV cache 的物理存储量**。核心逻辑在 slot mapping kernel（`compressor_utils.py:38`）：

```python
is_valid = (pos + 1) % COMPRESS_RATIO == 0    # 只有每第 N 个 token 获得 slot
slot_ids = tl.where(is_valid, slot_ids, PAD_ID)  # 其余丢弃
```

同时 `storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:327`），page 物理容量等比缩小。

从 V4 Flash 的 `compress_ratios` 配置和 KDA 替换后，40 个 V4 层的分布如下：

| 类型 | compress_ratio | 机制 | 存储 token 数 (2M 上下文) | 层数 |
|------|---------------|------|--------------------------|------|
| 全注意力 | 0 → clamp 为 1 | 每个 token 存储 | 2,097,152 | 2 |
| 压缩注意力 (C4A) | 4 | 每 4 个 token 存 1 个 | 2,097,152 / 4 = 524,288 | 21 |
| 滑窗注意力 (C128A) | 128 | 仅存最近 128 token | 128 | 17 |

#### 7.3.3 逐层计算（单请求, 2M context, FP8 KV cache）

**全注意力层（2 层, storage_block_size=256）：**
```
blocks = ceil(2,097,152 / 256) = 8,192
per_layer = 8,192 × (256 × 584 B) = 8,192 × 149,504 B = 1.14 GB
2 layers: 1.14 GB × 2 = 2.28 GB
```

**压缩注意力层（21 层, storage_block_size=64）：**
```
blocks = ceil(524,288 / 64) = 8,192
per_layer = 8,192 × (64 × 584 B) = 8,192 × 37,376 B = 0.285 GB
21 layers: 0.285 GB × 21 = 5.99 GB
```

**滑窗注意力层（17 层, storage_block_size=2）：**
```
blocks = ceil(128 / 2) = 64
per_layer = 64 × (2 × 584 B) = 64 × 1,168 B = 73 KB
17 layers: 73 KB × 17 ≈ 1.2 MB
```

**V4 KV cache 小计（单请求, 理论值）：**
```
2.28 GB + 5.99 GB + 0.001 GB ≈ 8.27 GB
```

加上 block table 元数据、对齐开销（约 10-15%）：单请求实际约 **9-10 GB**。

#### 7.3.4 KDA 层 Mamba State

KDA 层使用 recurrent state 替代 KV cache，**大小与上下文长度无关**：

| State 类型 | Shape | Elements | BF16 大小 |
|-----------|-------|----------|----------|
| Delta State | `(num_heads=32, head_dim=128, head_dim=128)` | 524,288 | 1.05 MB |
| Conv State | `(num_heads×head_dim=4096, conv_kernel_size=4)` | 16,384 | 0.03 MB |
| **单层合计** | | 540,672 | **1.08 MB** |
| **3 层合计 (FP8)** | | | **~1.6 MB** |

> mamba state 是**每序列固定大小**。1 token 和 2M token 的请求占用完全相同的 mamba cache。

#### 7.3.5 总览

| 组件 | 公式 | 单请求 (2M tokens) | 6 并发请求 |
|------|------|-------------------|-----------|
| 全注意力 KV (2 层) | 2 × 2M × 584 B | ~2.3 GB | ~13.7 GB |
| 压缩 KV (21 层) | 21 × 2M/4 × 584 B | ~6.0 GB | ~36.0 GB |
| 滑窗 KV (17 层) | 17 × 128 × 584 B | ~0.001 GB | ~0.007 GB |
| KV cache 理论值 | | **~8.3 GB** | **~49.7 GB** |
| KV cache + overhead | +10-15% | **~9.5 GB** | **~55 GB** |
| KDA mamba state (3 层) | 3 × 541K × 1 B | **~1.6 MB** | **~10 MB** |

> 原部署脚本的 "55 GB KV cache" 对应的是约 **6 个并发 2M 请求**的总 KV cache 量，而非单请求。单请求 2M 上下文的 KV cache 仅需约 **9.5 GB**。

### 7.4 总体内存预算

目标硬件：DGX Station GB300（1× Blackwell Ultra, **252 GB HBM3e**）

| 预算项 | 大小 | 占比 | 说明 |
|--------|------|------|------|
| 模型权重 (FP4 + BF16) | ~149 GB | 59% | 见 7.2 节 |
| KV cache (FP8, 6并发 × 2M) | ~55 GB | 22% | 见 7.3 节 |
| Mamba state (6并发) | ~10 MB | ~0% | 见 7.3.4 节 |
| CUDA graph, 中间激活, workspace | ~28 GB | 11% | |
| **已使用** | **~232 GB** | **92%** | |
| 余量 | ~20 GB | 8% | |

### 7.5 推荐启动参数

```bash
vllm serve <dk-checkpoint-path> \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype fp4 \
    --max-model-len 2097152 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --moe-backend deep_gemm_mega_moe \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --enforce-eager
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dtype fp4` | FP4 | 路由专家 NVFP4 量化，密集/注意力 BF16 |
| `--max-model-len` | 2097152 | 2M 上下文窗口 |
| `--gpu-memory-utilization` | 0.92 | 252 × 0.92 = 231.8 GB 可用 |
| `--kv-cache-dtype fp8` | FP8 | MLA KV cache 减半（vs BF16） |
| `--enable-prefix-caching` | 开启 | system prompt / tool 定义复用 |
| `--enable-expert-parallel` | 开启 | 256 专家 EP 并行 |
| `--moe-backend` | deep_gemm_mega_moe | Blackwell NVFP4 GEMM |
| `--speculative-config` | MTP, num_spec=1 | 投机解码，batch=1 下 TPOT 降低 30-50% |
| `--enforce-eager` | 开启 | 禁用 CUDA graph 以节省内存并避免编译开销 |

### 7.6 扩展建议

- **若需更大上下文**：可将 `gpu_memory_utilization` 提升至 0.94，释放额外 ~5 GB 给 KV cache，支持约 2.5M token（受 `max_model_len` 限制需同步调整）。
- **若需更低延迟**：可设置 `num_speculative_tokens: 2`，但需确认 MTP 第二层 head 的权重已正确映射。
- **多卡扩展**：如需更大 batch size 或更长上下文，可通过 `--tensor-parallel-size` 拆分到多张 GB300。

---

## 8. 测试策略

### 8.1 测试套件

| 测试 | 类型 | 覆盖范围 |
|------|------|---------|
| `test_projection_shapes` | 单元 | ProjectionIn/Out 张量维度验证 |
| `test_dk_config_defaults` | 单元 | DKConfig 序列化/反序列化 |
| `test_generate_weights` | 集成 | 权重生成 → 保存 → 重载 roundtrip |
| `test_tiny_model_inference` | 端到端 | 完整 vLLM 推理流程（需 H100/B200） |

### 8.2 测试 Checkpoint 生成

`scripts/generate_dk_test_weights.py` 生成一个极小的随机权重 checkpoint：
- 4-8 层，hidden_size=128-256，vocab=1000
- 无 MoE（1 expert = dense），无 FP4
- KDA 层在位置 [2, 4] 或 [3, 5, 7]
- 包含所有 DeepSeek V4 MLA 字段 + MHC 参数 + Kimi KDA 参数

### 8.3 已知限制

- **消费级 GPU 不支持**：MHC 的 DeepGEMM kernel 仅编译了数据中心 GPU（sm90+）的 compute capability。RTX 5090 等消费级 GPU 可以完成模型加载但在首次前向传播的 MHC 阶段失败。
- **需要 tilelang 依赖**：`mhc.py` 中的 TileLang kernel 在 CUDA 平台上为必需依赖，非 CUDA 平台跳过导入。

---

## 9. 依赖关系

```
DK Model 依赖:
├── DeepSeek V4 (deepseek_v4.py)
│   ├── MLA Attention + Compressor + Indexer
│   ├── MoE FFN (DeepGEMM MegaMoE on Blackwell)
│   └── MHC Pre/Post (mhc.py → TileLang + DeepGEMM)
├── Kimi Linear (kimi_linear.py)
│   ├── KimiGatedDeltaNetAttention (KDA)
│   └── KimiDecoderLayer (attention + FFN)
├── vLLM Engine Support
│   ├── HasInnerState (mamba state 管理)
│   ├── IsHybrid (KV cache + mamba state 混合)
│   └── SupportsPP (流水线并行)
└── 外部依赖
    ├── tilelang (MHC kernel JIT 编译)
    ├── DeepGEMM (tf32_hc_prenorm_gemm)
    └── safetensors (权重序列化)
```
