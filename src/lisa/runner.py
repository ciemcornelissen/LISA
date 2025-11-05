from __future__ import annotations

"""Runtime helpers for processing hyperspectral capture folders."""

from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import spectral as sp
from PIL import Image

from src.pipeline import (
    bunchTracker,
    data_finder,
    interpolate_gps,
    process_single_window,
)
from src.utilities import create_final_map_png_2

from .assets import PipelineAssets
from .config import PipelineRuntimeConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessedLog:
    """Simple helper to track which folders have been processed."""

    path: Path

    def read(self) -> Set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def append(self, folder_path: Path) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(str(folder_path.resolve()) + "\n")


def process_folder(
    folder_path: Path | str,
    assets: PipelineAssets,
    runtime: PipelineRuntimeConfig,
    use_legacy_rgb: bool = False,
    confidence: float = 0.4,
) -> bool:
    """Process a single acquisition folder and generate outputs.

    Args:
        folder_path: Capture folder containing a ``capture`` subfolder with ENVI files.
        assets: Preloaded models and cached tensors.
        runtime: Runtime configuration (window size, GPS range, etc.).
        use_legacy_rgb: If ``True`` uses the historic RGB scaling technique.
        confidence: Confidence threshold forwarded to the YOLO detector.

    Returns:
        ``True`` when the folder processed successfully, ``False`` otherwise.
    """

    folder_path = Path(folder_path)
    LOGGER.info("Processing capture folder %s", folder_path.name)

    try:
        data_locations = data_finder(str(folder_path))
        if not data_locations:
            LOGGER.warning("No ENVI files found in %s", folder_path)
            return False

        runtime.results_dir.mkdir(parents=True, exist_ok=True)

        hdr = sp.envi.open(data_locations["hdr"], data_locations["img"])
        hsi_data_cube = hdr.load()
        hsi_data_cube_rgb = hsi_data_cube.transpose(1, 0, 2)[::-1]

        if assets.config.get("dataLoader.normalize", True):
            mean_scaler = assets.images.scaler_images[0]
            std_scaler = assets.images.scaler_images[1]
            normalized_cube = (
                hsi_data_cube_rgb - mean_scaler[np.newaxis, np.newaxis, :]
            ) / std_scaler[np.newaxis, np.newaxis, :]
        else:
            normalized_cube = hsi_data_cube_rgb

        hsi_height, hsi_width, _ = normalized_cube.shape
        LOGGER.debug("HSI cube dimensions after preprocessing h=%s, w=%s", hsi_height, hsi_width)

        tracker = bunchTracker(match_threshold=runtime.tracker_match_threshold, slide_step=runtime.slide_step)
        all_results = []

        window_starts = range(0, max(hsi_width - runtime.window_width + 1, 1), runtime.slide_step)
        total_steps = len(window_starts)
        LOGGER.info("Sliding window across %s steps", total_steps)

        for index, start_x in enumerate(window_starts):
            end_x = min(start_x + runtime.window_width, hsi_width)
            LOGGER.debug("Processing window %s/%s pixels %s-%s", index + 1, total_steps, start_x, end_x)

            current_window = np.array(normalized_cube[:, start_x:end_x, :])
            current_window_rgb = np.array(hsi_data_cube_rgb[:, start_x:end_x, :])
            current_gps = interpolate_gps(runtime.start_point_gps, runtime.end_point_gps, index, total_steps)

            window_results = process_single_window(
                current_window,
                current_window_rgb,
                assets.models,
                tracker,
                current_gps,
                index,
                assets.images,
                start_x,
                assets.weight_config,
                assets.weight_images,
                rgbTechnique=use_legacy_rgb,
                confidence=confidence,
            )

            if window_results:
                all_results.extend(window_results)

        LOGGER.info("Detected %s grape bunches across all windows", len(all_results))

        rgb_bands = runtime.visualization.rgb_bands
        rgb_cube = hsi_data_cube_rgb[:, :, list(rgb_bands)]

        if use_legacy_rgb:
            histogram, bin_edges = np.histogram(rgb_cube, bins=runtime.visualization.histogram_bins)
            scale = bin_edges[-1] if bin_edges[-1] else 1.0
            rgb_cube = np.clip(rgb_cube / scale, 0.0, 1.0)
            rgb_image = np.sqrt(rgb_cube)
        else:
            percentile = np.percentile(rgb_cube, runtime.visualization.percentile_clip)
            percentile = percentile if percentile else rgb_cube.max() or 1.0
            rgb_cube = np.clip(rgb_cube / percentile, 0.0, 1.0)
            rgb_image = np.sqrt(rgb_cube)

        image_rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))

        output_path = runtime.results_dir / f"{folder_path.name}.png"
        create_final_map_png_2(
            image_rgb_pil,
            all_results,
            output_path=str(output_path),
            saveFig=runtime.visualization.save_fig,
            showFig=runtime.visualization.show_fig,
        )

        LOGGER.info("Saved annotated map to %s", output_path)
        return True

    except Exception:  # noqa: BLE001 - we want full traceback surfaced via logging
        LOGGER.exception("Failed to process folder %s", folder_path)
        return False


def _sorted_subfolders(parent: Path) -> Iterable[Path]:
    return sorted((child for child in parent.iterdir() if child.is_dir()), key=lambda item: item.stat().st_mtime)


def process_pending_folders(
    runtime: PipelineRuntimeConfig,
    assets: PipelineAssets,
    processed_log: ProcessedLog,
    known_processed: Set[str],
    use_legacy_rgb: bool = False,
    confidence: float = 0.4,
    update_log: bool = True,
) -> int:
    """Process all folders in ``runtime.watch_folder`` that are not yet logged."""

    success_counter = 0
    for folder in _sorted_subfolders(runtime.watch_folder):
        folder_key = str(folder.resolve())
        if folder_key in known_processed:
            continue
        if process_folder(folder, assets, runtime, use_legacy_rgb=use_legacy_rgb, confidence=confidence):
            success_counter += 1
            if update_log:
                processed_log.append(folder)
            known_processed.add(folder_key)
    return success_counter


def watch_for_new_folders(
    runtime: PipelineRuntimeConfig,
    assets: PipelineAssets,
    use_legacy_rgb: bool = False,
    confidence: float = 0.4,
    run_once: bool = False,
) -> None:
    """Continuously watch the configured folder for new captures."""

    runtime.ensure_directories()
    processed_log = ProcessedLog(runtime.processed_log_file)
    known_processed = processed_log.read()
    LOGGER.info("Found %s previously processed folders", len(known_processed))

    processed_in_bootstrap = process_pending_folders(
        runtime,
        assets,
        processed_log,
        known_processed,
        use_legacy_rgb=use_legacy_rgb,
        confidence=confidence,
        update_log=True,
    )
    LOGGER.info("Processed %s pending folders during startup", processed_in_bootstrap)

    if run_once:
        LOGGER.info("Run-once mode enabled; exiting watcher after initial sweep.")
        return

    try:
        while True:
            time.sleep(runtime.poll_interval_seconds)
            processed_now = process_pending_folders(
                runtime,
                assets,
                processed_log,
                known_processed,
                use_legacy_rgb=use_legacy_rgb,
                confidence=confidence,
                update_log=True,
            )
            if processed_now:
                LOGGER.info("Processed %s new folders", processed_now)
            else:
                LOGGER.debug("No new folders detected in %s", runtime.watch_folder)
    except KeyboardInterrupt:
        LOGGER.info("Watcher interrupted by user; shutting down.")


def process_single_directory(
    folder_path: Path | str,
    runtime: PipelineRuntimeConfig,
    assets: PipelineAssets,
    use_legacy_rgb: bool = False,
    confidence: float = 0.4,
    update_log: bool = False,
) -> bool:
    """Process a specific folder and optionally record it in the log file."""

    runtime.ensure_directories()
    processed_log = ProcessedLog(runtime.processed_log_file)
    known_processed = processed_log.read()

    folder = Path(folder_path)
    processed = process_folder(folder, assets, runtime, use_legacy_rgb=use_legacy_rgb, confidence=confidence)
    if processed and update_log:
        processed_log.append(folder)
        known_processed.add(str(folder.resolve()))
    return processed