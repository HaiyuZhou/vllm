# vllm/models/dk/model.py
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.models.dk.projection import ProjectionIn, ProjectionOut
from vllm.model_executor.models.kimi_linear import KimiDecoderLayer
from vllm.transformers_utils.configs.dk import DKConfig


class KimiKDALayer(nn.Module):
    """A KDA layer usable inside a DeepSeek V4 pipeline.

    Wraps a KimiDecoderLayer with input/output projections that handle the
    MHC multi-stream ↔ flat conversion.  Resets MHC state (residual,
    post_mix, res_mix) so the next DeepSeek layer does a standalone
    mhc_pre re-initialization.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DKConfig = vllm_config.model_config.hf_config
        self.hc_mult = config.hc_mult
        dv_hidden = config.hidden_size

        kimi_hidden = config.kimi_config.get("hidden_size", 4096)
        rms_norm_eps = config.kimi_config.get("rms_norm_eps", 1e-6)

        self.proj_in = ProjectionIn(
            self.hc_mult, dv_hidden, kimi_hidden,
            rms_norm_eps=rms_norm_eps, prefix=f"{prefix}.proj_in",
        )
        self.proj_out = ProjectionOut(
            self.hc_mult, dv_hidden, kimi_hidden,
            rms_norm_eps=rms_norm_eps, prefix=f"{prefix}.proj_out",
        )

        # Build a temporary KimiLinearConfig for the wrapped decoder layer.
        from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
        kimi_hf_config = KimiLinearConfig(**config.kimi_config)
        vllm_config.model_config.hf_config = kimi_hf_config
        self.kimi_layer = KimiDecoderLayer(
            config=kimi_hf_config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.kimi_layer",
        )
        vllm_config.model_config.hf_config = config

    def forward(
        self,
        x: torch.Tensor,              # (num_tokens, hc_mult, dv_hidden)
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, None, None]:
        # Flatten MHC multi-stream to single-stream via projection.
        flat_hidden = self.proj_in(x)  # (num_tokens, kimi_hidden)

        # KimiDecoderLayer expects (hidden_states, residual) pattern.
        # We always pass residual=None to start fresh inside the KDA layer.
        kimi_out, _kimi_residual = self.kimi_layer(
            positions=positions,
            hidden_states=flat_hidden,
            residual=None,
        )

        # Re-expand to MHC multi-stream.
        x_out = self.proj_out(kimi_out)  # (num_tokens, hc_mult, dv_hidden)

        # Reset MHC state so the next DeepSeek layer calls standalone mhc_pre.
        return x_out, None, None, None
