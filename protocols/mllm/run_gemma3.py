#!/usr/bin/env python3

from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from mllm_protocol import run_benchmark


MODEL_PATH = os.environ.get(
    "GEMMA3_MODEL_PATH",
    "/root/autodl-tmp/gemma3_models/models/AI-ModelScope--gemma-3-12b-it/snapshots/master",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

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
        generated = output[0, input_length:]
        return processor.decode(generated, skip_special_tokens=True).strip()

    run_benchmark(
        model_name="gemma3_12b_it",
        model_revision=MODEL_PATH,
        infer=infer,
        extra_config={"dtype": "bfloat16", "max_new_tokens": 96},
    )


if __name__ == "__main__":
    main()
