# vllm/models/dk/projection.py
import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm


class ProjectionIn(nn.Module):
    """Collapse MHC multi-stream to flat, then project to kimi hidden dim.

    Input:  (num_tokens, hc_mult, dv_hidden)
    Output: (num_tokens, kimi_hidden)
    """

    def __init__(self, hc_mult: int, dv_hidden: int, kimi_hidden: int,
                 rms_norm_eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.hc_mult = hc_mult
        in_features = hc_mult * dv_hidden
        self.proj = nn.Linear(in_features, kimi_hidden, bias=False)
        self.norm = RMSNorm(kimi_hidden, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_tokens, hc_mult, dv_hidden)
        x = x.flatten(1)  # (num_tokens, hc_mult * dv_hidden)
        x = self.proj(x)
        x = self.norm(x)
        return x


class ProjectionOut(nn.Module):
    """Project from kimi hidden dim back to MHC multi-stream.

    Input:  (num_tokens, kimi_hidden)
    Output: (num_tokens, hc_mult, dv_hidden)
    """

    def __init__(self, hc_mult: int, dv_hidden: int, kimi_hidden: int,
                 rms_norm_eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.hc_mult = hc_mult
        self.dv_hidden = dv_hidden
        out_features = hc_mult * dv_hidden
        self.proj = nn.Linear(kimi_hidden, out_features, bias=False)
        self.norm = RMSNorm(out_features, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_tokens, kimi_hidden)
        x = self.proj(x)
        x = self.norm(x)
        x = x.view(-1, self.hc_mult, self.dv_hidden)
        return x
