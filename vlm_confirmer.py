"""Reusable Qwen2.5-VL action confirmer for candidate security events."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from adapters.vlm_check import CLASSES

DEFAULT_MODEL = r"D:\Models\Qwen2.5-VL-3B-Instruct"


class VLMConfirmer:
    """Load one local VLM and classify a short sequence of BGR frames."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        gpu_memory_gib: float = 3.5,
        cpu_memory_gib: float = 5.0,
        max_pixels: int = 360 * 420,
    ):
        model_dir = Path(model_path)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Incomplete VLM model directory: {model_dir}")

        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self.processor.image_processor.max_pixels = max_pixels

        common = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs = {
                **common,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": quantization_config,
                "max_memory": {
                    0: f"{gpu_memory_gib:g}GiB",
                    "cpu": f"{cpu_memory_gib:g}GiB",
                },
            }
        else:
            kwargs = {
                **common,
                "torch_dtype": torch.bfloat16,
                "device_map": "cpu",
            }

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            **kwargs,
        )
        self.model.eval()

    @staticmethod
    def _prompt() -> str:
        return (
            "These images are chronological frames from one surveillance clip. "
            "Identify the main human action. Choose exactly one label from: "
            + ", ".join(CLASSES)
            + ". Use `normal` when no target action is present. "
            "`running` requires a clear fast running gait; ordinary movement "
            "is walking. `falling` requires a person moving to or lying on the "
            "ground. `fighting` requires aggressive physical interaction, not "
            "merely two people standing nearby. `kicking` requires a visible "
            "leg strike. `sitting` is seated posture. `waving` requires a "
            "raised hand moving side to side. `nodding` requires repeated "
            "down-and-up head movement, not a single bow. `handshaking` "
            "requires two people holding and moving joined hands. "
            "Use `uncertain` only when the frames are insufficient. "
            "Do not infer from filenames. Return only JSON with keys "
            "`action`, `confidence` (0 to 1), and `evidence`."
        )

    @staticmethod
    def _parse_answer(text: str, classes: list[str]) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                action = str(data.get("action", "parse_error")).strip().lower()
                if action not in classes and action != "uncertain":
                    action = "parse_error"
                confidence = float(data.get("confidence", 0) or 0)
                return {
                    "action": action,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "evidence": str(data.get("evidence", "")),
                    "raw_answer": text,
                }
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        return {
            "action": "parse_error",
            "confidence": 0.0,
            "evidence": text[:300],
            "raw_answer": text,
        }

    def classify(
        self,
        frames: list[np.ndarray],
        classes: list[str],
        prompt: str,
        max_new_tokens: int = 64,
    ) -> dict:
        if not frames:
            raise ValueError("No candidate frames supplied")

        images = [
            Image.fromarray(frame[:, :, ::-1].copy())
            for frame in frames
        ]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return self._parse_answer(answer, classes)

    def confirm(
        self,
        frames: list[np.ndarray],
        expected_action: str,
        threshold: float = 0.70,
    ) -> dict:
        parsed = self.classify(
            frames,
            classes=CLASSES,
            prompt=self._prompt(),
        )
        parsed["expected_action"] = expected_action
        parsed["accepted"] = (
            parsed["action"] == expected_action
            and parsed["confidence"] >= threshold
        )
        return parsed
