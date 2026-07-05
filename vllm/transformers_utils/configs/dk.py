# vllm/transformers_utils/configs/dk.py
from typing import Any

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


class DKConfig(DeepseekV4Config):
    """Configuration for the DK hybrid model.

    Extends DeepseekV4Config and accepts a kimi_config dict that carries
    the Kimi-Linear-48B sub-config.  The model_type is "dk".
    """

    model_type = "dk"

    def __init__(
        self,
        kda_layers: list[int] | None = None,
        kimi_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.kda_layers = kda_layers or [11, 21, 31]
        self.kimi_config = kimi_config or {}
        super().__init__(**kwargs)

    @property
    def is_hybrid(self) -> bool:
        return True

    @property
    def is_kda_layer_mask(self) -> list[bool]:
        """Return a bool mask: True for every layer idx that is KDA."""
        return [
            (i in self.kda_layers) for i in range(self.num_hidden_layers)
        ]
