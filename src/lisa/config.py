from __future__ import annotations

"""Configuration objects for the LISA pipeline CLI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class VisualizationConfig:
    """Controls how intermediate imagery and the final map are generated."""

    rgb_bands: Tuple[int, int, int] = (114, 58, 20)
    percentile_clip: float = 98.0
    histogram_bins: int = 10
    show_fig: bool = False
    save_fig: bool = True
    dpi: int = 300


@dataclass(slots=True)
class PipelineRuntimeConfig:
    """Holds runtime configuration for folder processing and watching."""

    watch_folder: Path | str = Path("data")
    processed_log_file: Path | str = Path("processed_folders.log")
    poll_interval_seconds: float = 10.0
    slide_step: int = 320
    window_width: int = 2048
    start_point_gps: Tuple[float, float] = (50.773636, 5.154737)
    end_point_gps: Tuple[float, float] = (50.773885, 5.154849)
    tracker_match_threshold: int = 75
    results_dir: Path | str = Path("results")
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        self.watch_folder = Path(self.watch_folder)
        self.processed_log_file = Path(self.processed_log_file)
        self.results_dir = Path(self.results_dir)

    def ensure_directories(self) -> None:
        """Ensure required directories exist before processing."""

        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.processed_log_file.exists():
            self.processed_log_file.touch()