#!/bin/bash
# deploy_gb300.sh — Launch DK model on DGX Station GB300
#
# Hardware: DGX Station GB300 (1× Blackwell Ultra, 252 GB HBM3e, 496 GB LPDDR5X)
# Model: DK Hybrid (DeepSeek-V4-Flash + Kimi KDA layers 11/21/31)
# Target: avg TPOT per user ≤ 20 ms at up to 2M context

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <dk-checkpoint-path>}"
PORT="${2:-8000}"

# Memory budget (HBM3e): ~148 GB weights + ~55 GB KV cache + ~49 GB headroom = 252 GB
# Use 0.92 utilization to leave breathing room for CUDA graphs and intermediates.
#
# FP4: DeepSeek experts via NVFP4, dense/attention/KDA in BF16.
# KV cache FP8: halves memory vs BF16.
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
