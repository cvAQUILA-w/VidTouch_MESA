# Zero-Shot MLLM Output Audit

The six RGB-only MLLMs use the same frozen candidate order and deterministic
prompt. Predictions must satisfy the task-specific JSON schema. One
format-only retry is allowed, and a remaining invalid response is scored as an
error. The exact prompt constructor, parser, retry rule, and evaluator are in
[`protocols/mllm`](../../../protocols/mllm/).

## Invalid Fabric-Level Outputs

| Model | Validation W/M/U/F | Test W/M/U/F |
|---|---|---|
| Gemma-3-12B-IT | 0/0/0/0 | 0/0/0/0 |
| GLM-4V-9B | 0/0/0/0 | 0/0/0/0 |
| InternVL2.5-8B | 0/0/0/0 | 0/0/0/0 |
| Llama-3.2-11B-Vision-Instruct | 0/0/4/0 | 0/0/3/0 |
| LLaVA-1.5-7B | 0/0/3/0 | 0/0/0/0 |
| Qwen2-VL-7B-Instruct | 0/0/1/0 | 0/0/0/0 |

## Valid but Class-Collapsed Material Predictions

| Model | Validation Macro | Test Macro | Test material classes with non-zero F1, out of 13 |
|---|---:|---:|---:|
| Gemma-3 | 14.95 | 15.78 | 3 |
| Llama-3.2 | 13.95 | 15.19 | 4 |
| Qwen2-VL | 18.15 | 10.73 | 2 |
| GLM-4V | 6.88 | 7.05 | 1 |
| InternVL2.5 | 6.70 | 6.70 | 1 |
| LLaVA-1.5 | 7.79 | 5.45 | 1 |

An invalid output is a schema or vocabulary failure. Class collapse instead
describes valid predictions concentrated in very few retained classes. The two
failure modes are counted separately in the released metrics.
