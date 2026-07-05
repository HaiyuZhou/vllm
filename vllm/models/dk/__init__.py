# vllm/models/dk/__init__.py
"""DK hybrid model — DeepSeek V4 + Kimi KDA, NVIDIA-only (GB300 target)."""

from .model import DKForCausalLM

__all__ = ["DKForCausalLM"]
