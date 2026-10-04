from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

MATERIAL_COMPONENT_LEXICON = (
    "polyamide",
    "polyester",
    "spandex",
    "viscose",
    "plastic",
    "cotton",
    "linen",
    "wool",
)

USAGE_SUPERCATEGORY = {
    "coat": "outerwear",
    "light-coat": "outerwear",
    "hoodie": "outerwear",
    "suit": "outerwear",
    "shirt": "upper-body",
    "t-shirt": "upper-body",
    "top": "upper-body",
    "knitwear": "upper-body",
    "dress": "one-piece",
    "skirt": "lower-body",
    "lining": "component",
}


def split_material_components(material: str) -> tuple[str, ...]:
    remaining = material
    components: list[str] = []
    while remaining:
        token = next(
            (candidate for candidate in MATERIAL_COMPONENT_LEXICON if remaining.startswith(candidate)),
            None,
        )
        if token is None:
            return ()
        components.append(token)
        remaining = remaining[len(token) :]
    return tuple(components)


@dataclass(frozen=True)
class FabricRecord:
    fabric_id: str
    weave: str
    material: str
    usage: str
    features: tuple[str, ...]
    image_paths: tuple[str, ...]
    video_paths: tuple[str, ...]


def parse_fabric_id(path: Path) -> str:
    match = re.match(r"^([A-Za-z0-9]+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse fabric id from {path.name}")
    return match.group(1)


def parse_label_file(label_path: str | Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid label line {line_no}: {raw!r}")
            fabric_id, weave, material, usage, *features = parts
            labels[fabric_id] = {
                "fabric_id": fabric_id,
                "weave": weave,
                "material": material,
                "usage": usage,
                "features": tuple(features),
            }
    return labels


def _scan_by_id(folder: Path, exts: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not folder.exists():
        raise FileNotFoundError(folder)
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        fid = parse_fabric_id(path)
        result.setdefault(fid, []).append(str(path))
    return result


def scan_vidtouch(data_root: str | Path, min_pairs_per_fabric: int = 1) -> list[FabricRecord]:
    root = Path(data_root)
    labels = parse_label_file(root / "label.txt")
    images = _scan_by_id(root / "RGBs", IMAGE_EXTS)
    videos = _scan_by_id(root / "TACs", VIDEO_EXTS)
    records: list[FabricRecord] = []
    for fid, label in sorted(labels.items()):
        image_paths = tuple(images.get(fid, []))
        video_paths = tuple(videos.get(fid, []))
        if len(image_paths) < min_pairs_per_fabric or len(video_paths) < min_pairs_per_fabric:
            continue
        records.append(
            FabricRecord(
                fabric_id=fid,
                weave=label["weave"],
                material=label["material"],
                usage=label["usage"],
                features=tuple(label["features"]),
                image_paths=image_paths,
                video_paths=video_paths,
            )
        )
    if not records:
        raise RuntimeError(f"No valid VidTouch records found under {root}")
    return records


class LabelVocab:
    def __init__(self, records: list[FabricRecord], allowed_labels: dict[str, list[str]] | None = None) -> None:
        allowed_labels = allowed_labels or {}

        def select_labels(name: str, values: set[str]) -> list[str]:
            allowed = allowed_labels.get(name)
            if allowed is None:
                return sorted(values)
            return sorted(set(allowed))

        self.fabric_ids = sorted({r.fabric_id for r in records})
        self.weaves = select_labels("weave", {r.weave for r in records})
        self.materials = select_labels("material", {r.material for r in records})
        self.usages = select_labels("usage", {r.usage for r in records})
        self.features = select_labels("features", {feat for r in records for feat in r.features})
        self.fabric_to_idx = {v: i for i, v in enumerate(self.fabric_ids)}
        self.weave_to_idx = {v: i for i, v in enumerate(self.weaves)}
        self.material_to_idx = {v: i for i, v in enumerate(self.materials)}
        self.usage_to_idx = {v: i for i, v in enumerate(self.usages)}
        self.feature_to_idx = {v: i for i, v in enumerate(self.features)}
        self._initialize_semantic_labels()

    def _initialize_semantic_labels(self) -> None:
        parsed_materials = {
            material: split_material_components(material)
            for material in self.materials
        }
        unknown = [material for material, components in parsed_materials.items() if not components]
        if unknown:
            raise ValueError(f"Cannot decompose material labels: {unknown}")
        self.material_components = sorted(
            {component for components in parsed_materials.values() for component in components}
        )
        self.material_primaries = sorted({components[0] for components in parsed_materials.values()})
        self.usage_supercategories = sorted(
            {USAGE_SUPERCATEGORY.get(usage, "other") for usage in self.usages}
        )
        self.material_component_to_idx = {
            value: index for index, value in enumerate(self.material_components)
        }
        self.material_primary_to_idx = {
            value: index for index, value in enumerate(self.material_primaries)
        }
        self.usage_supercategory_to_idx = {
            value: index for index, value in enumerate(self.usage_supercategories)
        }

    def encode(self, record: FabricRecord) -> dict[str, Any]:
        feature_vec = torch.zeros(len(self.features), dtype=torch.float32)
        for feat in record.features:
            if feat in self.feature_to_idx:
                feature_vec[self.feature_to_idx[feat]] = 1.0
        material_idx = self.material_to_idx.get(record.material, -1)
        material_component_vec = torch.zeros(len(self.material_components), dtype=torch.float32)
        material_primary = -1
        if material_idx >= 0:
            components = split_material_components(record.material)
            for component in components:
                material_component_vec[self.material_component_to_idx[component]] = 1.0
            material_primary = self.material_primary_to_idx[components[0]]
        usage_idx = self.usage_to_idx.get(record.usage, -1)
        usage_supercategory = -1
        if usage_idx >= 0:
            group = USAGE_SUPERCATEGORY.get(record.usage, "other")
            usage_supercategory = self.usage_supercategory_to_idx[group]
        return {
            "fabric": self.fabric_to_idx[record.fabric_id],
            "weave": self.weave_to_idx.get(record.weave, -1),
            "material": material_idx,
            "usage": usage_idx,
            "features": feature_vec,
            "material_components": material_component_vec,
            "material_primary": material_primary,
            "material_semantic_valid": float(material_idx >= 0),
            "usage_supercategory": usage_supercategory,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "fabric_ids": self.fabric_ids,
            "weaves": self.weaves,
            "materials": self.materials,
            "usages": self.usages,
            "features": self.features,
            "material_components": self.material_components,
            "material_primaries": self.material_primaries,
            "usage_supercategories": self.usage_supercategories,
        }

    def material_semantic_layout(self) -> tuple[list[list[float]], list[int]]:
        component_matrix: list[list[float]] = []
        primary_indices: list[int] = []
        for material in self.materials:
            components = split_material_components(material)
            component_set = set(components)
            component_matrix.append(
                [float(component in component_set) for component in self.material_components]
            )
            primary_indices.append(self.material_primary_to_idx[components[0]])
        return component_matrix, primary_indices

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LabelVocab":
        obj = cls.__new__(cls)
        obj.fabric_ids = list(state["fabric_ids"])
        obj.weaves = list(state["weaves"])
        obj.materials = list(state["materials"])
        obj.usages = list(state["usages"])
        obj.features = list(state["features"])
        obj.fabric_to_idx = {v: i for i, v in enumerate(obj.fabric_ids)}
        obj.weave_to_idx = {v: i for i, v in enumerate(obj.weaves)}
        obj.material_to_idx = {v: i for i, v in enumerate(obj.materials)}
        obj.usage_to_idx = {v: i for i, v in enumerate(obj.usages)}
        obj.feature_to_idx = {v: i for i, v in enumerate(obj.features)}
        obj._initialize_semantic_labels()
        return obj

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "fabric": len(self.fabric_ids),
            "weave": len(self.weaves),
            "material": len(self.materials),
            "usage": len(self.usages),
            "features": len(self.features),
            "material_components": len(self.material_components),
            "material_primaries": len(self.material_primaries),
            "usage_supercategories": len(self.usage_supercategories),
        }


def build_common_label_filter(
    records: list[FabricRecord],
    cfg: dict[str, Any] | None,
    count_source: str = "train_records",
) -> tuple[dict[str, list[str]] | None, dict[str, Any] | None]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None, None
    min_counts = cfg.get("min_counts", {})
    if not isinstance(min_counts, dict):
        min_counts = {}

    def min_count(name: str, default: int) -> int:
        return int(min_counts.get(name, default))

    counters = {
        "weave": Counter(r.weave for r in records),
        "material": Counter(r.material for r in records),
        "usage": Counter(r.usage for r in records),
        "features": Counter(feat for r in records for feat in r.features),
    }
    thresholds = {
        "weave": min_count("weave", 1),
        "material": min_count("material", 1),
        "usage": min_count("usage", 1),
        "features": min_count("features", 1),
    }
    allowed = {
        name: sorted(label for label, count in counter.items() if count >= thresholds[name])
        for name, counter in counters.items()
    }
    state = {
        "enabled": True,
        "count_source": count_source,
        "min_counts": thresholds,
        "allowed_labels": allowed,
        "raw_label_counts": {name: dict(sorted(counter.items())) for name, counter in counters.items()},
        "kept_label_counts": {name: len(labels) for name, labels in allowed.items()},
    }
    return allowed, state


def build_record_partitions_from_ids(
    records: list[FabricRecord],
    partition_ids: dict[str, list[str]],
) -> tuple[dict[str, list[FabricRecord]], dict[str, Any]]:
    known_ids = {record.fabric_id for record in records}
    partition_sets = {name: set(ids) for name, ids in partition_ids.items()}
    seen: set[str] = set()
    for name, ids in partition_sets.items():
        overlap = seen & ids
        if overlap:
            raise ValueError(f"Fixed split partition {name} overlaps earlier partitions: {sorted(overlap)}")
        seen |= ids
    unknown = seen - known_ids
    if unknown:
        raise ValueError(f"Fixed split contains unknown fabric IDs: {sorted(unknown)}")
    missing = known_ids - seen
    if missing:
        raise ValueError(f"Fixed split omits fabric IDs: {sorted(missing)}")
    partition_records = {
        name: [record for record in records if record.fabric_id in ids]
        for name, ids in partition_sets.items()
    }
    state = {
        "mode": "fabric",
        "strategy": "fixed_file",
        "partitions": {name: sorted(ids) for name, ids in partition_sets.items()},
    }
    return partition_records, state


def build_record_splits_from_ids(
    records: list[FabricRecord],
    train_ids: list[str],
    val_ids: list[str],
) -> tuple[list[FabricRecord], list[FabricRecord], dict[str, Any]]:
    partitions, state = build_record_partitions_from_ids(
        records,
        {"train": train_ids, "val": val_ids},
    )
    state["train_ids"] = state["partitions"]["train"]
    state["val_ids"] = state["partitions"]["val"]
    return partitions["train"], partitions["val"], state


def make_splits(
    records: list[FabricRecord],
    mode: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    fabric_ids = [r.fabric_id for r in records]
    if mode == "fabric":
        ids = fabric_ids[:]
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_ratio))
        val_ids = sorted(ids[:n_val])
        train_ids = sorted(ids[n_val:])
        return train_ids, val_ids
    if mode == "sample":
        return sorted(fabric_ids), sorted(fabric_ids)
    raise ValueError(f"Unknown split mode: {mode}")


def _split_paths(paths: tuple[str, ...], val_ratio: float, rng: random.Random) -> tuple[tuple[str, ...], tuple[str, ...]]:
    items = list(paths)
    rng.shuffle(items)
    if len(items) <= 1:
        return tuple(items), tuple(items)
    n_val = max(1, round(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)
    val = tuple(sorted(items[:n_val]))
    train = tuple(sorted(items[n_val:]))
    return train, val


def build_record_splits(
    records: list[FabricRecord],
    mode: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[FabricRecord], list[FabricRecord], dict[str, Any]]:
    if mode == "fabric":
        train_ids, val_ids = make_splits(records, mode, val_ratio, seed)
        train_set = set(train_ids)
        val_set = set(val_ids)
        train_records = [r for r in records if r.fabric_id in train_set]
        val_records = [r for r in records if r.fabric_id in val_set]
        return train_records, val_records, {"mode": mode, "train_ids": train_ids, "val_ids": val_ids}
    if mode != "sample":
        raise ValueError(f"Unknown split mode: {mode}")

    train_records: list[FabricRecord] = []
    val_records: list[FabricRecord] = []
    split_state: dict[str, Any] = {"mode": mode, "fabrics": {}}
    for rec in records:
        rng = random.Random(f"{seed}:{rec.fabric_id}")
        train_images, val_images = _split_paths(rec.image_paths, val_ratio, rng)
        train_videos, val_videos = _split_paths(rec.video_paths, val_ratio, rng)
        train_records.append(
            FabricRecord(
                fabric_id=rec.fabric_id,
                weave=rec.weave,
                material=rec.material,
                usage=rec.usage,
                features=rec.features,
                image_paths=train_images,
                video_paths=train_videos,
            )
        )
        val_records.append(
            FabricRecord(
                fabric_id=rec.fabric_id,
                weave=rec.weave,
                material=rec.material,
                usage=rec.usage,
                features=rec.features,
                image_paths=val_images,
                video_paths=val_videos,
            )
        )
        split_state["fabrics"][rec.fabric_id] = {
            "train_images": list(train_images),
            "val_images": list(val_images),
            "train_videos": list(train_videos),
            "val_videos": list(val_videos),
        }
    return train_records, val_records, split_state


def load_image(path: str, image_size: int, training: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if training:
        image = TF.resize(image, [image_size + 32, image_size + 32], antialias=True)
        # torchvision RandomCrop uses python RNG; keep this local and simple.
        max_i = image.height - image_size
        max_j = image.width - image_size
        top = random.randint(0, max(0, max_i))
        left = random.randint(0, max(0, max_j))
        image = TF.crop(image, top, left, image_size, image_size)
        if random.random() < 0.5:
            image = TF.hflip(image)
    else:
        image = TF.resize(image, [image_size, image_size], antialias=True)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor


def _uniform_indices(total: int, frames: int) -> np.ndarray:
    if total <= 0:
        return np.zeros(frames, dtype=np.int64)
    if total >= frames:
        return np.linspace(0, total - 1, frames).round().astype(np.int64)
    base = np.arange(total, dtype=np.int64)
    pad = np.full(frames - total, total - 1, dtype=np.int64)
    return np.concatenate([base, pad], axis=0)


def load_video_cv2(path: str, frames: int, size: int) -> torch.Tensor:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video decoding. Install opencv-python-headless."
        ) from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(total, frames)
    images: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if images:
                images.append(images[-1].copy())
                continue
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        images.append(frame)
    cap.release()
    arr = np.stack(images, axis=0)
    tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor


def video_cache_file(cache_root: str | Path, video_path: str | Path, frames: int, size: int) -> Path:
    cache_dir = Path(cache_root) / f"video_f{frames}_s{size}"
    return cache_dir / f"{Path(video_path).name}.pt"


def load_video(
    path: str,
    frames: int,
    size: int,
    cache_root: str | Path | None = None,
) -> torch.Tensor:
    if cache_root:
        cache_path = video_cache_file(cache_root, path, frames, size)
        if cache_path.exists():
            tensor = torch.load(cache_path, map_location="cpu")
            if isinstance(tensor, dict):
                tensor = tensor["video"]
            return tensor.float()
    return load_video_cv2(path, frames, size)


def _choose_paths(
    paths: tuple[str, ...],
    count: int,
    training: bool,
    offset: int = 0,
    with_replacement: bool = True,
) -> tuple[str, ...]:
    if not paths:
        raise RuntimeError("Cannot sample from an empty path list")
    count = max(1, count)
    if training:
        if not with_replacement and count <= len(paths):
            return tuple(random.sample(paths, count))
        return tuple(random.choice(paths) for _ in range(count))
    return tuple(paths[(offset + i) % len(paths)] for i in range(count))


def _choose_paired_paths(
    image_paths: tuple[str, ...],
    video_paths: tuple[str, ...],
    count: int,
    training: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pair_count = min(len(image_paths), len(video_paths))
    if pair_count == 0:
        raise RuntimeError("Cannot sample paired views from an empty path list")
    if training:
        indices = [random.randrange(pair_count) for _ in range(max(1, count))]
    else:
        indices = [index % pair_count for index in range(max(1, count))]
    return (
        tuple(image_paths[index] for index in indices),
        tuple(video_paths[index] for index in indices),
    )


class VidTouchPairDataset(Dataset):
    def __init__(
        self,
        records: list[FabricRecord],
        vocab: LabelVocab,
        fabric_ids: list[str],
        image_size: int,
        video_size: int,
        video_frames: int,
        training: bool,
        split_mode: str,
        repeats_per_epoch: int = 1,
        deterministic_val_pairs: bool = True,
        image_views_per_sample: int = 1,
        video_views_per_sample: int = 1,
        pair_recombination: bool = True,
        sample_with_replacement: bool = True,
        video_cache_root: str | Path | None = None,
        val_fabric_sets: bool = False,
    ) -> None:
        self.records_by_id = {r.fabric_id: r for r in records}
        self.records = [self.records_by_id[fid] for fid in fabric_ids if fid in self.records_by_id]
        self.vocab = vocab
        self.image_size = image_size
        self.video_size = video_size
        self.video_frames = video_frames
        self.training = training
        self.split_mode = split_mode
        self.repeats_per_epoch = max(1, repeats_per_epoch)
        self.deterministic_val_pairs = deterministic_val_pairs
        self.image_views_per_sample = max(1, int(image_views_per_sample))
        self.video_views_per_sample = max(1, int(video_views_per_sample))
        self.pair_recombination = bool(pair_recombination)
        self.sample_with_replacement = bool(sample_with_replacement)
        if not self.pair_recombination and self.image_views_per_sample != self.video_views_per_sample:
            raise ValueError("Paired sampling requires equal image and video view counts")
        self.video_cache_root = str(video_cache_root) if video_cache_root else None
        self.val_fabric_sets = bool(val_fabric_sets)
        self.set_level = self.image_views_per_sample > 1 or self.video_views_per_sample > 1 or self.val_fabric_sets
        if not self.records:
            raise RuntimeError("Dataset split has no records")
        self.val_pairs: list[tuple[FabricRecord, str, str]] = []
        if not training:
            for rec in self.records:
                if self.set_level:
                    self.val_pairs.append((rec, rec.image_paths[0], rec.video_paths[0]))
                    continue
                if deterministic_val_pairs:
                    n = min(len(rec.image_paths), len(rec.video_paths))
                    for i in range(max(1, n)):
                        image = rec.image_paths[min(i, len(rec.image_paths) - 1)]
                        video = rec.video_paths[min(i, len(rec.video_paths) - 1)]
                        self.val_pairs.append((rec, image, video))
                else:
                    for image in rec.image_paths:
                        for video in rec.video_paths:
                            self.val_pairs.append((rec, image, video))

    def __len__(self) -> int:
        if self.training:
            return len(self.records) * self.repeats_per_epoch
        return len(self.val_pairs)

    def _select_train_pair(self, rec: FabricRecord) -> tuple[str, str]:
        image = random.choice(rec.image_paths)
        video = random.choice(rec.video_paths)
        return image, video

    def _load_image_views(self, image_paths: tuple[str, ...]) -> torch.Tensor:
        images = [load_image(path, self.image_size, self.training) for path in image_paths]
        if self.image_views_per_sample == 1:
            return images[0]
        return torch.stack(images, dim=0)

    def _load_video_views(self, video_paths: tuple[str, ...]) -> torch.Tensor:
        videos = [
            load_video(path, self.video_frames, self.video_size, self.video_cache_root)
            for path in video_paths
        ]
        if self.video_views_per_sample == 1:
            return videos[0]
        return torch.stack(videos, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            rec = self.records[index % len(self.records)]
            if self.pair_recombination:
                image_paths = _choose_paths(
                    rec.image_paths,
                    self.image_views_per_sample,
                    True,
                    with_replacement=self.sample_with_replacement,
                )
                video_paths = _choose_paths(
                    rec.video_paths,
                    self.video_views_per_sample,
                    True,
                    with_replacement=self.sample_with_replacement,
                )
            else:
                image_paths, video_paths = _choose_paired_paths(
                    rec.image_paths,
                    rec.video_paths,
                    self.image_views_per_sample,
                    True,
                )
        else:
            rec, image_path, video_path = self.val_pairs[index]
            if self.set_level:
                image_paths = _choose_paths(rec.image_paths, self.image_views_per_sample, False)
                video_paths = _choose_paths(rec.video_paths, self.video_views_per_sample, False)
            else:
                image_paths = (image_path,)
                video_paths = (video_path,)
        image = self._load_image_views(image_paths)
        video = self._load_video_views(video_paths)
        labels = self.vocab.encode(rec)
        return {
            "image": image,
            "video": video,
            "fabric_id": rec.fabric_id,
            "fabric": torch.tensor(labels["fabric"], dtype=torch.long),
            "weave": torch.tensor(labels["weave"], dtype=torch.long),
            "material": torch.tensor(labels["material"], dtype=torch.long),
            "usage": torch.tensor(labels["usage"], dtype=torch.long),
            "features": labels["features"],
            "material_components": labels["material_components"],
            "material_primary": torch.tensor(labels["material_primary"], dtype=torch.long),
            "material_semantic_valid": torch.tensor(
                labels["material_semantic_valid"], dtype=torch.float32
            ),
            "usage_supercategory": torch.tensor(
                labels["usage_supercategory"], dtype=torch.long
            ),
            "image_path": image_paths[0] if len(image_paths) == 1 else list(image_paths),
            "video_path": video_paths[0] if len(video_paths) == 1 else list(video_paths),
        }
