# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


class DKConfig(DeepseekV4Config):
    """Configuration for the DK hybrid model.

    Extends DeepseekV4Config with kda_layers (KDA layer positions)
    and kimi_config (Kimi-Linear sub-configuration). The model_type
    is "dk".
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
