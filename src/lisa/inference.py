from __future__ import annotations

"""Inference utilities extracted from the legacy research pipeline."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from ultralytics import YOLO

from lisa.legacy_models import (
    EncoderAE_3D,
    SavGolFilterGPU,
    SpectralPredictorWithTransformer,
)

LOGGER = logging.getLogger(__name__)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _ensure_materialized(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required model artefact: {path}. Run `bash scripts/download_models.sh` to fetch pretrained weights."
        )
    try:
        with path.open("rb") as handle:
            header = handle.read(len(LFS_POINTER_PREFIX))
    except OSError as exc:  # pragma: no cover - surface helpful context
        raise RuntimeError(f"Unable to read required artefact: {path}") from exc
    if header.startswith(LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer. Run `git lfs pull` or `bash scripts/download_models.sh` to fetch the real file."
        )


def load_models_data(models_root: str | Path = "models") -> Tuple[Dict[str, object], Dict[str, object], object, object, Dict[str, object]]:
    """Load YOLO, transformer models, and cached datasets from disk."""

    models_root = Path(models_root)

    weight_checkpoint = models_root / "cross_weight_predictor_epoch_99_R2_0.757_end_R20.610_runID_i0mijofs.pth"
    yolo_checkpoint = models_root / "yoloModel_wandb_5l8pg9sf_map50old_0.5658.pt"
    ae_checkpoint = models_root / "ae_predictor_epoch_150_R2_0.5951396226882935_runID_cd7605a0.pth"

    required_files = [
        models_root / "configs.pkl",
        models_root / "images.pkl",
        models_root / "weight_images.pkl",
        weight_checkpoint,
        yolo_checkpoint,
        ae_checkpoint,
    ]
    for artefact in required_files:
        _ensure_materialized(artefact)

    with (models_root / "configs.pkl").open("rb") as handle:
        loaded = pickle.load(handle)
    config = loaded["config"]
    model_config = loaded["model_config"]
    ae_config = loaded["ae_config"]

    weight_config_params = torch.load(weight_checkpoint, map_location=torch.device("cpu"))

    yolo_model = YOLO(str(yolo_checkpoint), task="detect")

    weight_model = SpectralPredictorWithTransformer(weight_config_params["model_config"])
    weight_model.load_state_dict(weight_config_params["weight_predictor_state_dict"])
    weight_model.eval()

    predictor = SpectralPredictorWithTransformer(model_config)
    autoencoder = EncoderAE_3D(ae_config)

    ae_state = torch.load(ae_checkpoint, map_location=torch.device("cpu"))
    predictor.load_state_dict(ae_state["sugar_predictor_state_dict"])
    predictor.eval()
    autoencoder.load_state_dict(ae_state["encoder_state_dict"])
    autoencoder.eval()

    models = {
        "yolo": yolo_model,
        "weight": weight_model,
        "quality": predictor,
        "encoder": autoencoder,
    }

    LOGGER.info("Loading cached datasets from %s", models_root)
    with (models_root / "weight_images.pkl").open("rb") as handle:
        weight_images = pickle.load(handle)
    with (models_root / "images.pkl").open("rb") as handle:
        images = pickle.load(handle)

    return models, weight_config_params, weight_images, images, config


def snv_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        mean = torch.mean(x)
        std = torch.clamp(torch.std(x), min=1e-6)
        return (x - mean) / std
    if x.ndim >= 3:
        dim_to_reduce = 0 if x.ndim == 3 else 1
        if x.shape[dim_to_reduce] <= 1:
            return x
        mean = torch.mean(x, dim=dim_to_reduce, keepdim=True)
        std = torch.clamp(torch.std(x, dim=dim_to_reduce, keepdim=True), min=1e-6)
        return (x - mean) / std
    LOGGER.warning("SNV tensor processing not implemented for ndim=%s", x.ndim)
    return x


def create_pseudo_rgb(hsi_window: np.ndarray, *, use_legacy_scaling: bool = True) -> np.ndarray:
    bands = (114, 58, 20)
    data_point = hsi_window[:, :, list(bands)]
    if use_legacy_scaling:
        histogram, bin_edges = np.histogram(data_point, bins=10)
        scale = bin_edges[-1] if bin_edges[-1] else 1.0
        normalized = np.clip(data_point / scale, 0.0, 1.0)
    else:
        p_low, p_high = np.percentile(data_point, (2, 98))
        denom = (p_high - p_low) or 1.0
        normalized = np.clip((data_point - p_low) / denom, 0.0, 1.0)
    return np.sqrt(normalized)


def extract_patches_from_bbox(
    hsi_window: np.ndarray,
    bbox: Sequence[float],
    *,
    patch_size: Tuple[int, int] = (8, 8),
    stride: int = 3,
    center_bias_factor: float = 0.5,
    max_patches: int = 50,
) -> List[np.ndarray]:
    x1, y1, x2, y2 = map(int, bbox)
    bunch_hsi_data = hsi_window[y1:y2, x1:x2, :]

    h, w, _ = bunch_hsi_data.shape
    patch_h, patch_w = patch_size

    center_x = w / 2
    center_y = h / 2
    center_w = w * center_bias_factor
    center_h = h * center_bias_factor
    center_x1, center_y1 = center_x - center_w / 2, center_y - center_h / 2
    center_x2, center_y2 = center_x + center_w / 2, center_y + center_h / 2

    patches: List[np.ndarray] = []
    rng = np.random.default_rng()
    samples_seen = 0
    for y in range(0, max(h - patch_h + 1, 1), stride):
        for x in range(0, max(w - patch_w + 1, 1), stride):
            patch_center_x = x + patch_w / 2
            patch_center_y = y + patch_h / 2
            is_center = center_x1 <= patch_center_x <= center_x2 and center_y1 <= patch_center_y <= center_y2
            if is_center or rng.random() < 0.33:
                patch = bunch_hsi_data[y : y + patch_h, x : x + patch_w, :]
                samples_seen += 1
                if len(patches) < max_patches:
                    patches.append(patch)
                else:
                    replace_index = rng.integers(0, samples_seen)
                    if replace_index < max_patches:
                        patches[replace_index] = patch

    return patches


def preprocess_and_predict_quality(
    patches: List[np.ndarray],
    quality_model: torch.nn.Module,
    models: Dict[str, object],
) -> Tuple[float, float, float] | None:
    if not patches:
        return None

    savgol_gpu_deriv1 = SavGolFilterGPU(window_length=5, deriv=1)
    preprocessed = [savgol_gpu_deriv1(torch.tensor(p.copy())) for p in patches]
    batch_tensor = torch.from_numpy(np.array(preprocessed)).float()
    batch_tensor = batch_tensor.permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        if models.get("encoder"):
            batch_tensor = models["encoder"](batch_tensor)[0]
        outputs = quality_model(batch_tensor)

    brix_preds, acid_preds, is_grape_logits = outputs
    grape_mask = (is_grape_logits > 0).squeeze()

    num_grape = torch.sum(grape_mask).item()
    total = grape_mask.numel()
    grape_percentage = (num_grape / total) * 100 if total else 0
    if grape_percentage < 30:
        LOGGER.debug("Rejecting bunch: grape percentage %.1f%% below threshold", grape_percentage)
        return None

    if torch.any(grape_mask):
        avg_brix = brix_preds[grape_mask].mean().item()
        avg_acid = acid_preds[grape_mask].mean().item()
        return avg_brix, avg_acid, grape_percentage
    LOGGER.debug("No grape patches classified by quality model")
    return None


def preprocess_and_predict_quality_onnx(
    patches: List[np.ndarray],
    quality_session,
    models: Dict[str, object],
) -> Tuple[float, float, float] | None:
    """ONNX Runtime variant of :func:`preprocess_and_predict_quality`."""

    if not patches:
        return None

    savgol_gpu_deriv1 = SavGolFilterGPU(window_length=5, deriv=1)
    preprocessed = [savgol_gpu_deriv1(torch.tensor(p.copy())) for p in patches]
    batch_tensor = torch.stack(preprocessed).permute(0, 3, 1, 2).contiguous()

    ort_inputs = {quality_session.get_inputs()[0].name: batch_tensor.numpy()}
    outputs = quality_session.run(None, ort_inputs)
    brix_preds, acid_preds, is_grape_logits = outputs

    grape_mask = (is_grape_logits > 0).squeeze()
    num_grape = np.sum(grape_mask)
    total = grape_mask.size
    grape_percentage = (num_grape / total) * 100 if total else 0
    if grape_percentage < 30:
        LOGGER.debug("Rejecting bunch: grape percentage %.1f%% below threshold", grape_percentage)
        return None

    if np.any(grape_mask):
        avg_brix = brix_preds[grape_mask].mean()
        avg_acid = acid_preds[grape_mask].mean()
        return float(avg_brix), float(avg_acid), float(grape_percentage)
    LOGGER.debug("No grape patches classified by ONNX quality model")
    return None


def process_single_window(
    hsi_window: np.ndarray,
    hsi_window_rgb: np.ndarray,
    models: Dict[str, object],
    tracker: "bunchTracker",
    gps_coords: Tuple[float, float],
    window_index: int,
    dataset,
    start_x: int,
    weight_config: Dict[str, object],
    weight_dataset,
    *,
    rgbTechnique: bool = True,
    confidence: float = 0.4,
) -> List[Dict[str, object]]:
    window_results: List[Dict[str, object]] = []
    window_width = hsi_window.shape[1]

    pseudo_rgb = create_pseudo_rgb(hsi_window_rgb, use_legacy_scaling=rgbTechnique)
    if pseudo_rgb.max() <= 1.0:
        pseudo_rgb = (pseudo_rgb * 255).astype(np.uint8)
    pseudo_rgb_img = Image.fromarray(pseudo_rgb, "RGB")

    detection_results = models["yolo"](pseudo_rgb_img, conf=confidence)

    bboxes_tensor = detection_results[0].boxes.xyxy
    bboxes_for_tracker = bboxes_tensor.cpu().numpy()

    tracked_objects = tracker.update(bboxes_for_tracker)

    for bunch_id, bbox in tracked_objects.items():
        if bunch_id in tracker.processed_ids:
            continue

        x1_local, y1_local, x2_local, y2_local = bbox
        global_bbox = [x1_local + start_x, y1_local, x2_local + start_x, y2_local]

        x1, y1, x2, y2 = map(int, bbox)
        bunch_hsi_data = hsi_window_rgb[y1:y2, x1:x2, :]

        if models.get("weight") is not None:
            with torch.no_grad():
                current = bunch_hsi_data
                if weight_config.get("dataLoader.normalize"):
                    scaler_images = weight_dataset.scaler_images
                    current = (
                        current - scaler_images[0][np.newaxis, np.newaxis, :]
                    ) / scaler_images[1][np.newaxis, np.newaxis, :]
                weight_input = cv2.resize(
                    np.array(current),
                    dsize=(weight_config["model.patch_size"], weight_config["model.patch_size"]),
                    interpolation=cv2.INTER_CUBIC,
                ).transpose(2, 0, 1)
                if weight_config.get("dataLoader.processTechnique") in [4, 5]:
                    weight_input = snv_tensor(torch.tensor(weight_input.copy()))
                savgol_gpu_deriv = SavGolFilterGPU(
                    window_length=weight_config["dataLoader.Savitzky_Golay_window"],
                    deriv=2 if weight_config.get("dataLoader.processTechnique") in [3, 5] else 1,
                )
                preprocessed_weight = savgol_gpu_deriv(weight_input)
                predicted_weight = models["weight"].forward(preprocessed_weight.unsqueeze(0).float()).item()
        else:
            predicted_weight = 0.0

        patches = extract_patches_from_bbox(hsi_window, bbox)
        quality_results = preprocess_and_predict_quality(patches, models["quality"], models)
        del patches

        if quality_results:
            avg_brix, avg_acid, grape_percentage = quality_results
            if getattr(dataset, "normalize", False):
                avg_brix = dataset.scaler_labels_brix.inverse_transform(np.array([[avg_brix]])).item()
                avg_acid = dataset.scaler_labels_acid.inverse_transform(np.array([[avg_acid]])).item()
                if weight_config.get("dataLoader.normalize"):
                    predicted_weight = weight_dataset.scaler_labels_weights.inverse_transform(
                        np.array([[predicted_weight]])
                    ).item()
            window_results.append(
                {
                    "bunch_id": bunch_id,
                    "coordinates": gps_coords,
                    "predicted_weight_g": predicted_weight,
                    "predicted_brix": avg_brix,
                    "predicted_acidity": avg_acid,
                    "global_bounding_box": global_bbox,
                    "grape_percentage": grape_percentage,
                }
            )
            tracker.processed_ids.add(bunch_id)
    return window_results


class bunchTracker:
    def __init__(self, max_disappeared: int = 3, match_threshold: int = 50, slide_step: int = 180) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.bboxes: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.processed_ids: set[int] = set()
        self.max_disappeared = max_disappeared
        self.match_threshold = match_threshold
        self.slide_step = slide_step

    def _get_centroids(self, bboxes: Iterable[Sequence[float]]) -> np.ndarray:
        return np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes])

    def update(self, bboxes: Iterable[Sequence[float]]) -> Dict[int, np.ndarray]:
        bboxes = list(bboxes)
        if not bboxes:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return {}

        input_centroids = self._get_centroids(bboxes)

        if not self.objects:
            for i, centroid in enumerate(input_centroids):
                self._register(centroid, np.array(bboxes[i]))
            return self.bboxes

        object_ids = list(self.objects.keys())
        last_centroids = np.array(list(self.objects.values()))
        predicted_centroids = last_centroids - np.array([self.slide_step, 0])

        distance_matrix = cdist(predicted_centroids, input_centroids)
        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distance_matrix[row, col] > self.match_threshold:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = np.array(bboxes[col])
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unmatched_rows = set(range(len(predicted_centroids))) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col], np.array(bboxes[col]))

        return {oid: bbox for oid, bbox in self.bboxes.items() if self.disappeared.get(oid, 0) == 0}

    def _register(self, centroid: np.ndarray, bbox: np.ndarray) -> None:
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.bboxes.pop(object_id, None)
        self.disappeared.pop(object_id, None)


def interpolate_gps(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    index: int,
    total_steps: int,
) -> Tuple[float, float]:
    if total_steps <= 1:
        return start_coords
    fraction = index / (total_steps - 1)
    lat = start_coords[0] + fraction * (end_coords[0] - start_coords[0])
    lon = start_coords[1] + fraction * (end_coords[1] - start_coords[1])
    return lat, lon


def data_finder(folder: str | Path) -> Dict[str, str]:
    folder = Path(folder) / "capture"
    mapping: Dict[str, str] = {}
    if not folder.is_dir():
        LOGGER.warning("Capture folder %s not found", folder)
        return mapping
    for file in folder.iterdir():
        file_type = file.name.split("_")[0]
        if file.suffix.lower() == ".hdr" and file_type != "DARKREF":
            mapping["hdr"] = str(file)
        if file.suffix.lower() == ".raw" and file_type != "DARKREF":
            mapping["img"] = str(file)
    return mapping


__all__ = [
    "bunchTracker",
    "create_pseudo_rgb",
    "data_finder",
    "extract_patches_from_bbox",
    "interpolate_gps",
    "load_models_data",
    "preprocess_and_predict_quality_onnx",
    "preprocess_and_predict_quality",
    "process_single_window",
    "snv_tensor",
]
