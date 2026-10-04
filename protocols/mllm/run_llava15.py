#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
LLAVA_REPO = os.environ.get("LLAVA_REPO", "/root/autodl-tmp/llava")
sys.path.insert(0, LLAVA_REPO)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from mllm_protocol import run_benchmark


MODEL_PATH = os.environ.get(
    "LLAVA15_MODEL_PATH", "/root/autodl-tmp/models/llava-v1.5-7b"
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=model_name,
        device="cuda",
    )
    model.eval()

    def infer(image_path: str, prompt_text: str) -> str:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
        conversation = conv_templates["vicuna_v1"].copy()
        conversation.append_message(
            conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        )
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=96,
                use_cache=True,
            )
        generated = output[0, input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    run_benchmark(
        model_name="llava15_7b",
        model_revision=MODEL_PATH,
        infer=infer,
        extra_config={"dtype": "float16", "max_new_tokens": 96},
    )


if __name__ == "__main__":
    main()
