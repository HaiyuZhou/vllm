#!/bin/bash
# deploy_gb300.sh — Launch DK model on DGX Station GB300
#
# Hardware: DGX Station GB300 (1× Blackwell Ultra, 252 GB HBM3e, 496 GB LPDDR5X)
# Model: DK Hybrid (DeepSeek-V4-Flash + Kimi KDA layers 11/21/31)
# Target: avg TPOT per user ≤ 20 ms at up to 2M context

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <dk-checkpoint-path>}"
PORT="${2:-8000}"

# Memory budget (HBM3e): 252 GB total @ 0.92 utilization = 231.8 GB usable
#   Model weights:  ~149 GB (FP4 experts ~131.5 GB + BF16 dense/attention ~17.3 GB)
#   KV cache:       ~55 GB (FP8 fp8_ds_mla, ~6 concurrent 2M requests)
#     Per-request breakdown (2M tokens, V4 MLA):
#       Full-attn (2 layers):   2 × 2M × 584 B = 2.3 GB
#       Compressed (21 layers): 21 × 2M/4 × 584 B = 6.0 GB (only every 4th token stored)
#       Sliding-window (17):    17 × 128 × 584 B = 1.2 MB (128-token cap)
#       Single request total: ~9.5 GB (incl. block overhead)
#   Mamba state:    ~10 MB (3 KDA layers × 541K elems, per-sequence fixed size)
#   CUDA/workspace: ~28 GB
#   Headroom:        ~19 GB
#
# Model composition (from real configs):
#   Source A: deepseek-ai/DeepSeek-V4-Flash (43 layers, hc_mult=4, 256 FP4 experts)
#   Source B: moonshotai/Kimi-Linear-48B-A3B-Instruct (27 layers, 20 KDA layers)
#   DK result: 40 V4 layers + 3 KDA layers at positions 11, 21, 31
#   Attention mix: 2 full-attn + 21 C4A (every 4th token cached) + 17 C128A (128-tok window)
#
# FP4: DeepSeek + Kimi routed experts via NVFP4, dense/attention in BF16.
# KV cache: fp8_ds_mla format (448B K_nope FP8 + 128B K_rope FP16 + 8B scale).
# Prefix caching: agentic coding reuses system prompts and tool definitions.
# MegaMoE + EP: required for the DeepGEMM FP4 path on Blackwell.
# MTP speculation: reduces effective TPOT by 30-50% at batch size 1.

exec vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype fp4 \
    --max-model-len 2097152 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --moe-backend deep_gemm_mega_moe \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --enforce-eager \
    "$@"
