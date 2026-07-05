#!/usr/bin/env python3
"""Generate a tiny DK model checkpoint with random weights for testing.

Creates a minimal DK model that fits on a single consumer GPU (e.g. RTX 5090).
The model uses 8 layers with KDA at positions 3, 5, 7, no MoE (dense FFN only),
tiny hidden dimensions, and a small vocabulary.

Usage:
    python scripts/generate_dk_test_weights.py --output-path ./dk-test-checkpoint
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def build_tiny_config(args) -> dict:
    """Build a minimal DK config with all required fields for the V4 model."""
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    kda_layers = args.kda_layers

    return {
        # === DK-specific ===
        "model_type": "dk",
        "architectures": ["DKForCausalLM"],
        "kda_layers": kda_layers,
        "kimi_config": {
            "model_type": "kimi_linear",
            "torch_dtype": "bfloat16",
            "hidden_size": hidden_size,
            "num_attention_heads": args.num_heads,
            "num_key_value_heads": args.num_kv_heads,
            "head_dim": hidden_size // args.num_heads,
            "intermediate_size": hidden_size * 4,
            "num_hidden_layers": num_layers,
            "vocab_size": args.vocab_size,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": True,
            "rope_theta": 10000.0,
            "linear_attn_config": {
                "kda_layers": [1, 3, 5],       # 1-indexed, first 3 KDA layers
                "full_attn_layers": [2, 4, 6, 7, 8],
                "num_heads": args.num_heads,
                "head_dim": hidden_size // args.num_heads,
                "short_conv_kernel_size": 4,
            },
            "first_k_dense_replace": 0,
            "moe_layer_freq": 0,                # No MoE in Kimi part
        },

        # === DeepSeek V4 base ===
        "hidden_size": hidden_size,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "head_dim": hidden_size // args.num_heads,
        "intermediate_size": hidden_size * 4,
        "moe_intermediate_size": hidden_size,
        "num_hidden_layers": num_layers,
        "vocab_size": args.vocab_size,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "max_position_embeddings": args.max_seq_len,
        "rope_theta": 10000.0,

        # === DeepSeek V4 MHC ===
        "hc_mult": args.hc_mult,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "hc_post_alpha": 2.0,

        # === DeepSeek V4 MoE (minimal — 1 expert = dense-like) ===
        "n_routed_experts": 1,
        "n_shared_experts": 0,
        "num_experts_per_tok": 1,
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
        "scoring_func": "softmax",
        "num_hash_layers": 0,
        "first_k_dense_replace": num_layers,    # All layers are "dense"
        "moe_layer_freq": 0,
        "swiglu_limit": None,
        "expert_dtype": "bf16",                  # No FP4 for consumer GPU
        "enable_eplb": False,
        "eplb_config": {},

        # === DeepSeek V4 MLA attention fields ===
        "q_lora_rank": hidden_size // 2,
        "o_lora_rank": hidden_size,
        "qk_rope_head_dim": hidden_size // args.num_heads,
        "o_groups": 1,
        "sliding_window": None,
        "compress_ratios": [1] * num_layers,
        "compress_rope_theta": 10000.0,
        "torch_dtype": "bfloat16",

        # === DeepSeek V4 sparse attention (minimal) ===
        "index_topk": 32,
        "index_heads": args.num_heads,
        "index_head_dim": hidden_size // args.num_heads,
        "compress_ratio": 4,
        "rope_scaling": None,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "quantization_config": {"scale_fmt": "ue8m0"},  # Required by V4 attention init
    }


def generate_weights(config: dict) -> dict[str, torch.Tensor]:
    """Generate random BF16 weights for every parameter the DK model expects.

    Walks the expected parameter names based on the config and generates
    appropriately shaped random tensors.
    """
    weights: dict[str, torch.Tensor] = {}
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    hc = config["hc_mult"]
    vocab = config["vocab_size"]
    intermediate = config["intermediate_size"]
    kda_layers = config["kda_layers"]
    kimi_h = config["kimi_config"]["hidden_size"]
    kimi_n_heads = config["kimi_config"]["num_attention_heads"]
    kimi_intermediate = config["kimi_config"]["intermediate_size"]

    def r(shape, name=""):
        """Random BF16 tensor, small init scale."""
        return torch.randn(shape, dtype=torch.bfloat16) * 0.02

    # --- Embedding (HF naming: mapper converts embed.weight → model.embed_tokens.weight) ---
    weights["embed.weight"] = r((vocab, h))

    for layer_idx in range(n_layers):
        pfx = f"model.layers.{layer_idx}.layer."

        if layer_idx + 1 in kda_layers:
            # === KDA layer ===
            # ProjectionIn
            in_feat = hc * h
            weights[pfx + "proj_in.proj.weight"] = r((kimi_h, in_feat))
            weights[pfx + "proj_in.norm.weight"] = torch.ones(kimi_h, dtype=torch.bfloat16)
            # ProjectionOut
            weights[pfx + "proj_out.proj.weight"] = r((in_feat, kimi_h))
            weights[pfx + "proj_out.norm.weight"] = torch.ones(in_feat, dtype=torch.bfloat16)

            # KDA attention (KimiGatedDeltaNetAttention)
            kap = pfx + "kimi_layer.self_attn."
            weights[kap + "A_log"] = r((kimi_n_heads,))
            weights[kap + "D_log"] = r((kimi_n_heads,))
            weights[kap + "dt_bias"] = r((kimi_n_heads,))
            weights[kap + "gate_proj.weight"] = r((kimi_n_heads * head_dim, kimi_h)) * 0.02
            weights[kap + "q_proj.weight"] = r((kimi_n_heads * head_dim, kimi_h))
            weights[kap + "k_proj.weight"] = r((kimi_n_heads * head_dim, kimi_h))
            weights[kap + "v_proj.weight"] = r((kimi_n_heads * head_dim, kimi_h))
            weights[kap + "o_proj.weight"] = r((kimi_h, kimi_n_heads * head_dim))
            # Short conv (causal conv1d)
            for proj in ["q", "k", "v"]:
                weights[kap + f"{proj}_conv1d.weight"] = r(
                    (kimi_n_heads * head_dim, 1, 4)
                )
                weights[kap + f"{proj}_conv1d.bias"] = r((kimi_n_heads * head_dim,))
            # Output norm
            weights[kap + "norm.weight"] = torch.ones(kimi_h, dtype=torch.bfloat16)

            # KDA FFN (dense MLP for test — no MoE)
            kf_prefix = pfx + "kimi_layer."
            weights[kf_prefix + "mlp.gate_up_proj.weight"] = r((2 * kimi_intermediate, kimi_h))
            weights[kf_prefix + "mlp.down_proj.weight"] = r((kimi_h, kimi_intermediate))
            weights[kf_prefix + "input_layernorm.weight"] = torch.ones(kimi_h, dtype=torch.bfloat16)
            weights[kf_prefix + "post_attention_layernorm.weight"] = torch.ones(kimi_h, dtype=torch.bfloat16)

        else:
            # === DeepSeek V4 layer ===
            # Attention
            attn = pfx + "attn."
            # Q projections
            weights[attn + "fused_wqa_wkv.weight"] = r((
                n_heads * (head_dim + head_dim) + n_kv_heads * head_dim,
                h,
            ))
            weights[attn + "q_norm.weight"] = torch.ones(n_heads * head_dim, dtype=torch.bfloat16)
            weights[attn + "k_norm.weight"] = torch.ones(n_kv_heads * head_dim, dtype=torch.bfloat16)
            # KV compressor params
            weights[attn + "kv_a_proj_with_mqa.weight"] = r((head_dim + head_dim, h))
            weights[attn + "kv_a_layernorm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
            weights[attn + "kv_b_proj.weight"] = r((n_heads * (head_dim + head_dim), head_dim))
            # Output
            weights[attn + "o_proj.weight"] = r((h, n_heads * head_dim))
            # Indexer (DeepseekV4Indexer — sparse attention top-k selector)
            index_n_heads = n_heads
            q_lora_rank = h  # simplified: use hidden_size as q_lora_rank for tiny model
            weights[attn + "indexer.wq_b.weight"] = r((index_n_heads * head_dim, q_lora_rank))
            weights[attn + "indexer.weights_proj.weight"] = r((index_n_heads, h))
            # indexer.compressor
            weights[attn + "indexer.compressor.kv_score_bias"] = torch.zeros(1, dtype=torch.bfloat16)
            weights[attn + "indexer.compressor.wgate.weight"] = r((1, n_heads * head_dim))
            weights[attn + "indexer.compressor.wkv.weight"] = r((n_kv_heads * head_dim, n_heads * head_dim))
            # Attention's own compressor (C4A KV compression, separate from indexer's)
            weights[attn + "compressor.kv_score_bias"] = torch.zeros(1, dtype=torch.bfloat16)
            weights[attn + "compressor.wgate.weight"] = r((1, n_heads * head_dim))
            weights[attn + "compressor.wkv.weight"] = r((n_kv_heads * head_dim, n_heads * head_dim))

            # FFN (1 expert = dense-like)
            moe = pfx + "ffn."
            weights[moe + "gate.weight"] = r((1, h))
            weights[moe + "experts.w13_weight"] = r((1, 2 * intermediate, h))
            weights[moe + "experts.w2_weight"] = r((1, h, intermediate))

            # Layer norms
            weights[pfx + "attn_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            weights[pfx + "ffn_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            # MHC params
            mix_hc = (2 + hc) * hc
            hc_dim = hc * h
            weights[pfx + "hc_attn_fn"] = torch.randn((mix_hc, hc_dim), dtype=torch.float32) * 0.02
            weights[pfx + "hc_ffn_fn"] = torch.randn((mix_hc, hc_dim), dtype=torch.float32) * 0.02
            weights[pfx + "hc_attn_base"] = torch.randn((mix_hc,), dtype=torch.float32) * 0.02
            weights[pfx + "hc_ffn_base"] = torch.randn((mix_hc,), dtype=torch.float32) * 0.02
            weights[pfx + "hc_attn_scale"] = torch.randn((3,), dtype=torch.float32) * 0.02
            weights[pfx + "hc_ffn_scale"] = torch.randn((3,), dtype=torch.float32) * 0.02

    # Final norm
    weights["model.norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    # MHC head
    hc_dim = hc * h
    weights["model.hc_head_fn"] = torch.randn((hc, hc_dim), dtype=torch.float32) * 0.02
    weights["model.hc_head_base"] = torch.randn((hc,), dtype=torch.float32) * 0.02
    weights["model.hc_head_scale"] = torch.randn((1,), dtype=torch.float32) * 0.02
    # LM head (tied with embeddings)
    # LM head (HF naming: mapper converts head.weight → lm_head.weight)
    weights["head.weight"] = r((vocab, h))

    return weights


def main():
    parser = argparse.ArgumentParser(description="Generate tiny DK test checkpoint")
    parser.add_argument("--output-path", default="./dk-test-checkpoint")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--kda-layers", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--hc-mult", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_path
    if os.path.exists(output_dir):
        if not args.force:
            print(f"Output path {output_dir!r} already exists. Use --force to overwrite.")
            sys.exit(1)
    else:
        os.makedirs(output_dir, exist_ok=True)

    config = build_tiny_config(args)
    weights = generate_weights(config)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Save weights
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(weights, weights_path)
    print(f"Weights saved to {weights_path} ({len(weights)} tensors)")

    # Summary
    total_params = sum(w.numel() for w in weights.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 2 / 1e9:.2f} GB (BF16)")
    print(f"\nKDA layers at positions: {args.kda_layers}")
    print(f"DeepSeek V4 layers at other positions: 0-{args.num_layers - 1}")
    print(f"\nTo test with vllm:")
    print(f"  vllm serve {output_dir} --dtype bfloat16 --enforce-eager --max-model-len 2048")


if __name__ == "__main__":
    main()
