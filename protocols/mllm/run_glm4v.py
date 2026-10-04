#!/usr/bin/env python3

from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm_protocol import run_benchmark


MODEL_PATH = os.environ.get(
    "GLM4V_MODEL_PATH",
    "/root/autodl-tmp/glm_models/models/ZhipuAI--glm-4v-9b/snapshots/master",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to("cuda")

    def infer(image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
                use_cache=True,
            )
        return tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True
        ).strip()

    run_benchmark(
        model_name="glm4v_9b",
        model_revision=MODEL_PATH,
        infer=infer,
        extra_config={"dtype": "bfloat16", "max_new_tokens": 96},
    )


if __name__ == "__main__":
    main()
