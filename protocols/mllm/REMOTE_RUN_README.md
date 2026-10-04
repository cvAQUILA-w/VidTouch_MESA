# VidTouch MLLM Benchmark: Remote Run

## Start after switching the instance to GPU mode

```bash
cd /root/VBTSINT_DATASET/benchmark/protocol_v2
bash launch_benchmark.sh
```

The launcher starts a detached `screen` session named `vidtouch_mllm_v2`.

## Monitor

```bash
screen -ls
tail -f /root/VBTSINT_DATASET/benchmark/protocol_v2/outputs/formal_v1/master.log
```

Attach to the interactive session only when needed:

```bash
screen -r vidtouch_mllm_v2
```

Detach without stopping it by pressing `Ctrl+A`, then `D`.

## Resume

Run the same launch command again after an interruption. Completed Fabric IDs are
read from JSONL and skipped. Do not delete partially completed output files.

## MLLM-Fabric Llama checkpoint

No fine-tuned Llama checkpoint or adapter was found on this server on 2026-07-18.
Without additional settings, the launcher evaluates the base
Llama-3.2-11B-Vision-Instruct and labels it accordingly.

For a PEFT/LoRA checkpoint:

```bash
export LLAMA32_ADAPTER_PATH=/absolute/path/to/mllm-fabric-adapter
cd /root/VBTSINT_DATASET/benchmark/protocol_v2
bash launch_benchmark.sh
```

For a fully merged fine-tuned checkpoint:

```bash
export LLAMA32_MODEL_PATH=/absolute/path/to/merged-model
cd /root/VBTSINT_DATASET/benchmark/protocol_v2
bash launch_benchmark.sh
```

Never label the base checkpoint as MLLM-Fabric.

## Outputs

```text
/root/VBTSINT_DATASET/benchmark/protocol_v2/outputs/formal_v1/
```

Each model directory contains:

- `config.json`
- `val_predictions.jsonl`
- `val_metrics.json`
- `test_predictions.jsonl`
- `test_metrics.json`

The final cross-model table is written to `summary.md`.

InternVL2.5 was previously stored under volatile `/dev/shm` and is currently
missing. The launcher runs the five persistent models first, then downloads
InternVL2.5 through ModelScope and runs it. Set `AUTO_DOWNLOAD_INTERNVL=0` and
`INTERNVL25_MODEL_PATH` to use a manually prepared copy.
