# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for DK model loading."""
import json
import os
import tempfile

import pytest
import torch

from vllm.transformers_utils.configs.dk import DKConfig
from vllm.models.dk.projection import ProjectionIn, ProjectionOut


def test_projection_shapes():
    hcm, dv, kimi = 4, 2048, 4096
    pin = ProjectionIn(hcm, dv, kimi)
    pout = ProjectionOut(hcm, dv, kimi)

    x = torch.randn(2, hcm, dv)
    y = pin(x)
    assert y.shape == (2, kimi)
    z = pout(y)
    assert z.shape == (2, hcm, dv)


def test_dk_config_defaults():
    config = DKConfig(
        num_hidden_layers=40, hidden_size=2048,
        kda_layers=[11, 21, 31],
        kimi_config={"hidden_size": 4096, "linear_attn_config": {
            "kda_layers": [5, 15, 25],
            "full_attn_layers": list(range(32)),
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        }},
    )
    assert config.model_type == "dk"
    assert config.kda_layers == [11, 21, 31]
    assert config.is_kda_layer_mask[11] is True
    assert config.is_kda_layer_mask[10] is False
    assert config.is_hybrid is True


def test_generate_weights():
    """Test that the weight generation script produces valid checkpoint."""
    from scripts.generate_dk_test_weights import build_tiny_config, generate_weights
    from safetensors.torch import save_file

    # Use a tiny namespace for args
    class Args:
        hidden_size = 128
        num_layers = 4
        kda_layers = [2, 4]
        num_heads = 4
        num_kv_heads = 2
        hc_mult = 2
        vocab_size = 100
        max_seq_len = 512

    config = build_tiny_config(Args)

    # Verify config structure
    assert config["model_type"] == "dk"
    assert config["kda_layers"] == [2, 4]
    assert "kimi_config" in config
    assert config["kimi_config"]["linear_attn_config"]["kda_layers"]

    # Generate weights
    weights = generate_weights(config)

    # Verify key parameters exist
    assert "model.embed_tokens.weight" in weights, "Missing embedding"
    assert "lm_head.weight" in weights, "Missing LM head"
    assert "model.norm.weight" in weights, "Missing final norm"

    # Verify KDA layers have projections
    assert "model.layers.1.proj_in.proj.weight" in weights, "Missing KDA proj_in at layer 2"
    assert "model.layers.1.proj_out.proj.weight" in weights, "Missing KDA proj_out at layer 2"

    # Verify DeepSeek layers have MHC params
    assert "model.layers.0.hc_attn_fn" in weights, "Missing MHC at layer 1"
    assert "model.layers.0.hc_ffn_fn" in weights, "Missing MHC at layer 1"

    # Save and reload roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Verify saved files exist
        assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
        assert os.path.exists(config_path)

        # Verify saved config is valid JSON and loads back
        with open(config_path) as f:
            reloaded = json.load(f)
        assert reloaded["model_type"] == "dk"

    print(f"Generated {len(weights)} tensors, total params: "
          f"{sum(w.numel() for w in weights.values()):,}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tiny_model_inference():
    """End-to-end: generate checkpoint, load with vllm, run inference.

    This test generates a tiny DK checkpoint with random weights, loads it
    via vLLM, and verifies that inference produces output tokens.  Designed
    to run on a single consumer GPU (e.g. RTX 5090, ~24+ GB VRAM).

    Run:
        pytest tests/models/test_dk.py::test_tiny_model_inference -v -s
    """
    from scripts.generate_dk_test_weights import build_tiny_config, generate_weights
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams

    class Args:
        hidden_size = 256
        num_layers = 4
        kda_layers = [2, 4]
        num_heads = 4
        num_kv_heads = 2
        hc_mult = 2
        vocab_size = 1000
        max_seq_len = 512

    config = build_tiny_config(Args)
    weights = generate_weights(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        llm = LLM(
            model=tmpdir,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.5,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
        )
        outputs = llm.generate(["Hello, world!"], sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0
        print(f"Generated {len(outputs[0].outputs[0].token_ids)} tokens: "
              f"{outputs[0].outputs[0].text}")
