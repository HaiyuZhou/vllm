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


class DKDecoderLayer(nn.Module):
    """Dispatch between DeepSeek V4 and Kimi KDA layers by layer index."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

        config: DKConfig = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)

        self.is_kda = layer_idx in config.kda_layers

        if self.is_kda:
            self.layer = KimiKDALayer(vllm_config, prefix=prefix)
        else:
            self.layer = DeepseekV4DecoderLayer(
                vllm_config, prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.layer(x, positions, input_ids, post_mix, res_mix, residual)

from vllm.model_executor.models.interfaces import (
    HasInnerState, IsHybrid, MixtureOfExperts, SupportsPP,
)
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import (
    PPMissingLayer, make_layers, maybe_prefix,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc, MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator, MambaStateShapeCalculator,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from itertools import islice


class DKModel(nn.Module):
    """DK hybrid model — DeepSeek V4 base with KDA layers at positions 11, 21, 31."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DKConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda pfx: DKDecoderLayer(
                vllm_config, prefix=pfx,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty((self.hc_mult, self.hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty((self.hc_mult,), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype, device=device,
            ),
        })

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

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        residual, post_mix, res_mix = None, None, None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, input_ids,
                post_mix, res_mix, residual,
            )

        # Collapse multi-stream to flat for the norm+head.
        # If the last layer was KDA, residual/post_mix/res_mix are all None
        # and we collapse by taking the mean over the hc_mult dimension.
        if residual is not None:
            hidden_states = mhc_post_tilelang(
                hidden_states, residual, post_mix, res_mix,
            )
        else:
            hidden_states = hidden_states.mean(dim=1)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        num_tokens = hidden_states.shape[0]
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
            self.rms_norm_eps, self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DKForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    """Top-level DK model for vLLM serving."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = DKModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size, self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs,
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.kimi_config["linear_attn_config"]["num_heads"],
            hf_config.kimi_config["linear_attn_config"]["head_dim"],
            conv_kernel_size=hf_config.kimi_config["linear_attn_config"]["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.utils import is_pp_missing_parameter
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
