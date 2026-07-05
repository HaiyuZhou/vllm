# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for DK model loading."""
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
