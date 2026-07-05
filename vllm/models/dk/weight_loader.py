#!/usr/bin/env python3
"""Assemble DK model checkpoint from DeepSeek-V4-Flash and Kimi-Linear-48B.

Usage:
    python vllm/models/dk/weight_loader.py \
        --deepseek-path deepseek-ai/DeepSeek-V4-Flash \
        --kimi-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --output-path ./dk-checkpoint \
        --kda-layers 11,21,31
"""

import argparse
import json
import os
import shutil


def load_config(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def load_safetensors(path: str) -> dict:
    """Load all safetensors from a directory into a single weights dict."""
    from safetensors.torch import load_file

    weights = {}
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".safetensors"):
                weights.update(load_file(os.path.join(path, fname)))
    elif path.endswith(".safetensors"):
        weights = load_file(path)
    return weights


def find_kda_layer_indices(kimi_config: dict) -> list[int]:
    """Return the 1-indexed KDA layer indices from the Kimi config."""
    la = kimi_config.get("linear_attn_config") or {}
    return la.get("kda_layers", [])


def remap_kda_weight(name: str, dk_layer_idx: int, kimi_kda_idx: int) -> str | None:
    """Remap a Kimi weight name to DK naming for a specific layer.

    Kimi naming: model.layers.{kimi_kda_idx}.{rest}
    DK naming:   model.layers.{dk_layer_idx}.kimi_layer.{rest}
    """
    prefix = f"model.layers.{kimi_kda_idx}."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    return f"model.layers.{dk_layer_idx}.kimi_layer.{rest}"


def main():
    import torch
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description="Assemble DK model checkpoint")
    parser.add_argument("--deepseek-path", required=True)
    parser.add_argument("--kimi-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--kda-layers", default="11,21,31",
                        help="Comma-separated 1-indexed KDA layer positions")
    args = parser.parse_args()

    dk_kda_layers = [int(x) for x in args.kda_layers.split(",")]
    os.makedirs(args.output_path, exist_ok=True)

    # --- Load source configs ---
    ds_config = load_config(args.deepseek_path)
    kimi_config = load_config(args.kimi_path)

    kimi_kda_indices = find_kda_layer_indices(kimi_config)
    if len(kimi_kda_indices) < len(dk_kda_layers):
        raise ValueError(
            f"Kimi model has {len(kimi_kda_indices)} KDA layers but "
            f"DK needs {len(dk_kda_layers)}. Not enough KDA layers to copy."
        )
    # Take the first N KDA layers from Kimi.
    kda_mapping = dict(zip(dk_kda_layers, kimi_kda_indices[:len(dk_kda_layers)]))

    # --- Build DK config ---
    ds_config["model_type"] = "dk"
    ds_config["kda_layers"] = dk_kda_layers
    ds_config["kimi_config"] = kimi_config
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(ds_config, f, indent=2)

    # --- Load weights ---
    print("Loading DeepSeek weights...")
    ds_weights = load_safetensors(args.deepseek_path)
    print(f"  {len(ds_weights)} tensors")

    print("Loading Kimi weights...")
    kimi_weights = load_safetensors(args.kimi_path)
    print(f"  {len(kimi_weights)} tensors")

    # --- Build output weight dict ---
    output_weights: dict = {}

    # 1. Start with all DeepSeek weights.
    for name, tensor in ds_weights.items():
        output_weights[name] = tensor

    # 2. Remove DeepSeek weights for the KDA layers and replace with Kimi weights.
    for dk_layer_idx, kimi_kda_idx in kda_mapping.items():
        # Remove DeepSeek weights for this layer.
        ds_prefix = f"model.layers.{dk_layer_idx - 1}."
        removed = [k for k in output_weights if k.startswith(ds_prefix)]
        for k in removed:
            del output_weights[k]
        print(f"  Removed {len(removed)} DeepSeek weights for layer {dk_layer_idx}")

        # Add Kimi KDA weights for this layer.
        added = 0
        for name, tensor in kimi_weights.items():
            new_name = remap_kda_weight(name, dk_layer_idx - 1, kimi_kda_idx - 1)
            if new_name is not None:
                output_weights[new_name] = tensor
                added += 1
        print(f"  Added {added} Kimi weights for DK layer {dk_layer_idx}")

    # 3. Initialize projection weights (random normal, small scale).
    kimi_hidden = kimi_config["hidden_size"]
    dv_hidden = ds_config["hidden_size"]
    hc_mult = ds_config.get("hc_mult", 4)
    in_features = hc_mult * dv_hidden

    for dk_layer_idx in dk_kda_layers:
        layer_prefix = f"model.layers.{dk_layer_idx - 1}."
        proj_in_name = layer_prefix + "proj_in.proj.weight"
        proj_out_name = layer_prefix + "proj_out.proj.weight"

        output_weights[proj_in_name] = torch.randn(kimi_hidden, in_features) * 0.02
        output_weights[proj_out_name] = torch.randn(in_features, kimi_hidden) * 0.02
        print(f"  Initialized projection weights for layer {dk_layer_idx} "
              f"(in: {kimi_hidden}x{in_features}, out: {in_features}x{kimi_hidden})")

    # --- Save ---
    print(f"Saving {len(output_weights)} tensors to {args.output_path}...")
    save_file(output_weights, os.path.join(args.output_path, "model.safetensors"))

    # Copy tokenizer from DeepSeek.
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(args.deepseek_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_path, fname))
            print(f"  Copied {fname}")

    print("Done.")


if __name__ == "__main__":
    main()
