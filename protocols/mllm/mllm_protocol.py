#!/usr/bin/env python3
"""Shared inference protocol for VidTouch RGB-only MLLM baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from evaluate_benchmark import evaluate


PROTOCOL_VERSION = "vidtouch_mllm_rgb_multiview_v1"
DATASET_ROOT = Path(os.environ.get("VIDTOUCH_DATASET_ROOT", "/root/VBTSINT_DATASET"))
SPLIT_PATH = DATASET_ROOT / "benchmark/splits/fabric_common_v2.json"
LABEL_PATH = DATASET_ROOT / "label.txt"
RGB_DIR = DATASET_ROOT / "RGBs"
OUTPUT_ROOT = Path(
    os.environ.get(
        "VIDTOUCH_BENCHMARK_OUTPUT",
        str(DATASET_ROOT / "benchmark/protocol_v2/outputs/formal_v1"),
    )
)
TASKS = ("weave", "material", "usage", "features")
TASK_NAMES = {
    "weave": "weave pattern",
    "material": "ordered material composition",
    "usage": "usage",
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allowed_labels() -> tuple[dict[str, Any], dict[str, list[str]]]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return split, split["metadata"]["label_filter"]["allowed_labels"]


def find_rgb_paths(fabric_id: str) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [
        path
        for path in RGB_DIR.iterdir()
        if path.is_file()
        and path.suffix.casefold() in suffixes
        and (path.stem == fabric_id or path.name.startswith(f"{fabric_id} "))
    ]
    paths.sort(key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No RGB image found for Fabric ID {fabric_id}")
    return paths


def task_prompt(task: str, candidates: list[str], retry: bool = False) -> str:
    candidate_text = ", ".join(candidates)
    prefix = (
        "The previous response did not follow the required JSON schema. "
        if retry
        else ""
    )
    if task == "features":
        return (
            f"{prefix}Classify the visible fabric from this RGB image only. "
            "Select every applicable functional feature from the fixed candidate list. "
            "An empty list is allowed. Return exactly one JSON object with no Markdown "
            'and no explanation: {"labels":["candidate1","candidate2"]}. '
            f"Use only exact strings from this fixed list: {candidate_text}"
        )
    return (
        f"{prefix}Classify the visible fabric's {TASK_NAMES[task]} from this RGB "
        "image only. Select exactly one candidate. Return exactly one JSON object "
        'with no Markdown and no explanation: {"label":"one exact candidate"}. '
        f"Use only an exact string from this fixed list: {candidate_text}"
    )


def _candidate_json_strings(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = []
    for match in reversed(list(_JSON_FENCE_RE.finditer(stripped))):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1].strip())

    candidates.append(stripped)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def parse_response(
    response: str, task: str, candidates: list[str]
) -> tuple[Any, bool]:
    lookup = {candidate.casefold(): candidate for candidate in candidates}

    for candidate_text in _candidate_json_strings(response):
        try:
            payload = json.loads(candidate_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if task == "features":
            if set(payload) != {"labels"} or not isinstance(payload["labels"], list):
                continue
            parsed: list[str] = []
            for value in payload["labels"]:
                if not isinstance(value, str):
                    break
                label = lookup.get(value.strip().casefold())
                if label is None:
                    break
                if label not in parsed:
                    parsed.append(label)
            else:
                return parsed, True
            continue
        if set(payload) != {"label"} or not isinstance(payload["label"], str):
            continue
        label = lookup.get(payload["label"].strip().casefold())
        if label is not None:
            return label, True
    return None, False


def infer_task(
    infer: Callable[[str, str], str],
    image_path: Path,
    task: str,
    candidates: list[str],
) -> tuple[Any, bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for retry in (False, True):
        prompt = task_prompt(task, candidates, retry=retry)
        started = time.perf_counter()
        response = infer(str(image_path), prompt)
        elapsed = time.perf_counter() - started
        prediction, valid = parse_response(response, task, candidates)
        attempts.append(
            {
                "retry": retry,
                "prompt": prompt,
                "response": response,
                "elapsed_sec": elapsed,
                "valid": valid,
            }
        )
        if valid:
            return prediction, True, attempts
    return None, False, attempts


def load_completed(path: Path, model_name: str, partition: str) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("protocol") != PROTOCOL_VERSION:
            raise ValueError(f"{path}:{line_number}: protocol mismatch")
        if record.get("model") != model_name:
            raise ValueError(f"{path}:{line_number}: model mismatch")
        if record.get("partition") != partition:
            raise ValueError(f"{path}:{line_number}: partition mismatch")
        fabric_id = record.get("fabric_id")
        if fabric_id in completed:
            raise ValueError(f"{path}:{line_number}: duplicate Fabric ID {fabric_id}")
        completed.add(fabric_id)
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_config(
    model_dir: Path,
    model_name: str,
    model_revision: str,
    extra_config: dict[str, Any] | None,
) -> None:
    split, allowed = load_allowed_labels()
    config = {
        "protocol": PROTOCOL_VERSION,
        "model": model_name,
        "model_revision": model_revision,
        "input": "all RGB views, independent inference, fabric-level deterministic vote",
        "partitions": ["val", "test"],
        "split_file": str(SPLIT_PATH),
        "split_file_sha256": file_sha256(SPLIT_PATH),
        "assignment_sha256": split["metadata"]["assignment_sha256"],
        "labels_file": str(LABEL_PATH),
        "labels_file_sha256": file_sha256(LABEL_PATH),
        "allowed_labels": allowed,
        "decoding": {
            "do_sample": False,
            "temperature": 0,
            "format_retry_count": 1,
        },
        "prompt_templates": {
            task: task_prompt(task, allowed[task]) for task in TASKS
        },
    }
    if extra_config:
        config["model_config"] = extra_config
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Existing config differs at {path}; use a new output directory"
            )
    else:
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_benchmark(
    model_name: str,
    model_revision: str,
    infer: Callable[[str, str], str],
    extra_config: dict[str, Any] | None = None,
) -> None:
    split, allowed = load_allowed_labels()
    model_dir = OUTPUT_ROOT / model_name
    write_config(model_dir, model_name, model_revision, extra_config)
    print(f"Protocol: {PROTOCOL_VERSION}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Output: {model_dir}", flush=True)

    for partition in ("val", "test"):
        predictions_path = model_dir / f"{partition}_predictions.jsonl"
        metrics_path = model_dir / f"{partition}_metrics.json"
        completed = load_completed(predictions_path, model_name, partition)
        partition_ids = split[f"{partition}_ids"]
        print(
            f"{partition}: {len(completed)}/{len(partition_ids)} Fabric IDs complete",
            flush=True,
        )
        for index, fabric_id in enumerate(partition_ids, start=1):
            if fabric_id in completed:
                continue
            views = []
            image_paths = find_rgb_paths(fabric_id)
            for image_index, image_path in enumerate(image_paths, start=1):
                parsed_prediction: dict[str, Any] = {}
                valid_output: dict[str, bool] = {}
                raw_responses: dict[str, Any] = {}
                for task in TASKS:
                    prediction, valid, attempts = infer_task(
                        infer, image_path, task, allowed[task]
                    )
                    parsed_prediction[task] = prediction
                    valid_output[task] = valid
                    raw_responses[task] = attempts
                views.append(
                    {
                        "image_path": str(image_path.relative_to(DATASET_ROOT)),
                        "view_index": image_index,
                        "parsed_prediction": parsed_prediction,
                        "valid_output": valid_output,
                        "raw_responses": raw_responses,
                    }
                )
                print(
                    f"[{partition} {index}/{len(partition_ids)}] "
                    f"{fabric_id} view {image_index}/{len(image_paths)}",
                    flush=True,
                )
            append_jsonl(
                predictions_path,
                {
                    "protocol": PROTOCOL_VERSION,
                    "partition": partition,
                    "model": model_name,
                    "model_revision": model_revision,
                    "seed": 0,
                    "fabric_id": fabric_id,
                    "view_predictions": views,
                },
            )

        result = evaluate(
            predictions_path=predictions_path,
            labels_path=LABEL_PATH,
            split_path=SPLIT_PATH,
            partition=partition,
        )
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{partition}: Macro Main={100 * result['metrics']['macro_main']:.2f}, "
            f"Legacy Main={100 * result['metrics']['legacy_main']:.2f}",
            flush=True,
        )
