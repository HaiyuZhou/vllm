# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DK hybrid model — DeepSeek V4 base with KDA layers at positions 11, 21, 31."""

import re
from collections.abc import Iterable
from itertools import islice
from typing import ClassVar, Literal

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v4 import (
    DeepseekV4DecoderLayer,
    hc_head,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.kimi_linear import KimiDecoderLayer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.dk import DKConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.multi_stream_utils import AuxStreamType


class ProjectionIn(nn.Module):
    """Collapse MHC multi-stream to flat, then project to Kimi hidden dim.

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
        x = x.flatten(1)
        x = self.proj(x)
        x = self.norm(x)
        return x


class ProjectionOut(nn.Module):
    """Project from Kimi hidden dim back to MHC multi-stream.

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
        x = self.proj(x)
        x = self.norm(x)
        x = x.view(-1, self.hc_mult, self.dv_hidden)
        return x


class KimiKDALayer(nn.Module):
    """A KDA layer inside a DK pipeline.

    Projects MHC multi-stream to flat single-stream, runs a KimiDecoderLayer
    (with KDA attention), then projects back. Each enclosing DeepSeek V4
    layer manages MHC independently, so no state is passed across boundaries.
    """

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""):
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

        # Build a KimiLinearConfig from the kimi_config sub-dict.
        kimi_cfg = KimiLinearConfig(**config.kimi_config)

        # Map DK KDA layer position → Kimi KDA layer index (0-indexed).
        # config.kda_layers  = [11, 21, 31]  (1-indexed DK positions)
        # kimi_cfg.linear_attn_config["kda_layers"] = [5, 15, 25] (1-idx Kimi)
        kda_layers_dk = config.kda_layers
        kda_layers_kimi = kimi_cfg.linear_attn_config["kda_layers"]
        dk_pos = layer_idx + 1  # 1-indexed
        kimi_kda_pos = kda_layers_kimi[kda_layers_dk.index(dk_pos)]  # 1-indexed
        kimi_layer_idx = kimi_kda_pos - 1  # 0-indexed

        self.kimi_layer = KimiDecoderLayer(
            config=kimi_cfg,
            layer_idx=kimi_layer_idx,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            parallel_config=vllm_config.parallel_config,
            model_config=vllm_config.model_config,
            prefix=f"{prefix}.kimi_layer",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                input_ids: torch.Tensor | None) -> torch.Tensor:
        # x: (num_tokens, hc_mult, dv_hidden) — MHC multi-stream
        flat = self.proj_in(x)                       # → (num_tokens, kimi_hidden)
        kimi_out, _ = self.kimi_layer(positions, flat, None)
        x_out = self.proj_out(kimi_out)              # → (num_tokens, hc_mult, dv_hidden)
        return x_out


class DKDecoderLayer(nn.Module):
    """Dispatch between DeepSeek V4 decoder layers and Kimi KDA layers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        config: DKConfig = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)

        self.is_kda = (layer_idx + 1) in config.kda_layers

        if self.is_kda:
            self.layer = KimiKDALayer(vllm_config, layer_idx, prefix=prefix)
        else:
            self.layer = DeepseekV4DecoderLayer(
                vllm_config, prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_dict=aux_stream_dict,
            )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.layer(x, positions, input_ids)


class DKModel(nn.Module):
    """Inner DK model — MHC expansion, layer dispatch, hc_head collapse.

    Dispatches layers through DKDecoderLayer instead of DeepseekV4DecoderLayer.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DKConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(1)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
        }

        self.device = current_platform.device_type
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DKDecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_dict=self.aux_stream_dict,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        self._mtp_hidden_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.hc_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions, input_ids)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse DeepseekV4Model's weight loading logic for the DeepSeek layers,
        # augmented with Kimi projection and KDA layer loading.
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DKForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    """Top-level DK model for vLLM serving.

    Implements hybrid-model interfaces so the engine can manage KV cache
    for DeepSeek V4 layers and mamba state for KDA layers.
    """

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True
    supports_pp: ClassVar[Literal[True]] = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex={
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DKConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = DKModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds,
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.config.hc_mult, self.config.hidden_size),
                dtype=dtype, device=device,
            ),
        })

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        parallel_config = vllm_config.parallel_config
        hf_config: DKConfig = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config else 0
        )
        kimi_cfg = hf_config.kimi_config
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            kimi_cfg["linear_attn_config"]["num_heads"],
            kimi_cfg["linear_attn_config"]["head_dim"],
            conv_kernel_size=kimi_cfg["linear_attn_config"]["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
