"""Inspect the Qwen2.5-VL model architecture without loading 7 GB of weights.

The architecture is the Transformers class
`Qwen2_5_VLForConditionalGeneration`, which contains:
    - a Qwen2.5 language model
    - a Qwen2.5-VL vision transformer
    - a multimodal projector into the language embedding space

This script constructs the model on PyTorch "meta" tensors, so it is fast and
uses no GPU/CPU weight memory.

Run:
    python vlm_architecture.py
    python vlm_architecture.py --print-modules

Outputs:
    vlm_architecture.json
    vlm_architecture.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = r"D:\Models\Qwen2.5-VL-3B-Instruct"


def get_attr(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Qwen2.5-VL architecture")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--print-modules", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model)
    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(config)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    vision = get_attr(config, "vision_config", {}) or {}
    text = get_attr(config, "text_config", None)

    summary = {
        "model_dir": str(model_dir),
        "architecture": get_attr(config, "architectures", []),
        "model_type": get_attr(config, "model_type"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "text_model": {
            "hidden_size": get_attr(text or config, "hidden_size"),
            "num_hidden_layers": get_attr(text or config, "num_hidden_layers"),
            "num_attention_heads": get_attr(text or config, "num_attention_heads"),
            "num_key_value_heads": get_attr(text or config, "num_key_value_heads"),
            "intermediate_size": get_attr(text or config, "intermediate_size"),
            "vocab_size": get_attr(text or config, "vocab_size"),
            "max_position_embeddings": get_attr(
                text or config, "max_position_embeddings"
            ),
            "hidden_act": get_attr(text or config, "hidden_act"),
            "rope_scaling": get_attr(text or config, "rope_scaling"),
        },
        "vision_model": {
            "depth": get_attr(vision, "depth"),
            "hidden_size": get_attr(vision, "hidden_size"),
            "intermediate_size": get_attr(vision, "intermediate_size"),
            "num_heads": get_attr(vision, "num_heads"),
            "patch_size": get_attr(vision, "patch_size"),
            "spatial_merge_size": get_attr(vision, "spatial_merge_size"),
            "temporal_patch_size": get_attr(vision, "temporal_patch_size"),
            "out_hidden_size": get_attr(vision, "out_hidden_size"),
            "tokens_per_second": get_attr(vision, "tokens_per_second"),
            "fullatt_block_indexes": get_attr(vision, "fullatt_block_indexes"),
        },
        "multimodal_tokens": {
            "vision_start_token_id": get_attr(config, "vision_start_token_id"),
            "vision_end_token_id": get_attr(config, "vision_end_token_id"),
            "image_token_id": get_attr(config, "image_token_id"),
            "video_token_id": get_attr(config, "video_token_id"),
        },
    }
    json_path = model_dir / "vlm_architecture.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    text_path = model_dir / "vlm_architecture.txt"
    text_path.write_text(str(model), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {text_path}")
    if args.print_modules:
        print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
