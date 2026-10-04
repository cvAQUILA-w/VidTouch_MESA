#!/usr/bin/env python3

from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from mllm_protocol import run_benchmark


BASE_PATH = os.environ.get(
    "LLAMA32_BASE_PATH",
    "/root/autodl-tmp/llama32_models/Llama-3.2-11B-Vision-Instruct",
)
MODEL_PATH = os.environ.get("LLAMA32_MODEL_PATH", BASE_PATH)
ADAPTER_PATH = os.environ.get("LLAMA32_ADAPTER_PATH")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if ADAPTER_PATH:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_name = (
        "mllm_fabric_llama32_11b"
        if ADAPTER_PATH or MODEL_PATH != BASE_PATH
        else "llama32_11b_vision_instruct_base"
    )
    revision = f"base={MODEL_PATH};adapter={ADAPTER_PATH or 'none'}"

    def infer(image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, text, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
                use_cache=True,
            )
        return processor.decode(
            output[0, input_length:], skip_special_tokens=True
        ).strip()

    run_benchmark(
        model_name=model_name,
        model_revision=revision,
        infer=infer,
        extra_config={
            "dtype": "bfloat16",
            "max_new_tokens": 96,
            "base_path": BASE_PATH,
            "model_path": MODEL_PATH,
            "adapter_path": ADAPTER_PATH,
        },
    )


if __name__ == "__main__":
    main()
