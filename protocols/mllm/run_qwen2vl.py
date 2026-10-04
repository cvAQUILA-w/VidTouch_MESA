#!/usr/bin/env python3

from __future__ import annotations

import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from mllm_protocol import run_benchmark


MODEL_PATH = os.environ.get(
    "QWEN2VL_MODEL_PATH",
    "/root/autodl-tmp/qwen_models/Qwen2-VL-7B-Instruct",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    def infer(image_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
                use_cache=True,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    run_benchmark(
        model_name="qwen2vl_7b_instruct",
        model_revision=MODEL_PATH,
        infer=infer,
        extra_config={"dtype": "bfloat16", "max_new_tokens": 96},
    )


if __name__ == "__main__":
    main()
