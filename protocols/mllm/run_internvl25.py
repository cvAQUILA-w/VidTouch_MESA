#!/usr/bin/env python3

from __future__ import annotations

import os

import torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel, AutoTokenizer

from mllm_protocol import run_benchmark


MODEL_PATH = os.environ.get(
    "INTERNVL25_MODEL_PATH",
    "/dev/shm/modelscope_cache/models/OpenGVLab--InternVL2_5-8B/snapshots/master",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    transform = T.Compose(
        [
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def infer(image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = (
            transform(image).unsqueeze(0).to("cuda").to(torch.bfloat16)
        )
        return model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config={
                "do_sample": False,
                "max_new_tokens": 96,
            },
        ).strip()

    run_benchmark(
        model_name="internvl25_8b",
        model_revision=MODEL_PATH,
        infer=infer,
        extra_config={
            "dtype": "bfloat16",
            "image_size": 448,
            "max_new_tokens": 96,
        },
    )


if __name__ == "__main__":
    main()
