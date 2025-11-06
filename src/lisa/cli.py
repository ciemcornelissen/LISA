from __future__ import annotations

"""Command line interface for the LISA grape quality pipeline."""

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .assets import PipelineAssets
from .config import PipelineRuntimeConfig, VisualizationConfig
from .runner import process_single_directory, watch_for_new_folders

DEFAULT_RGB_BANDS = (114, 58, 20)
DEFAULT_START_GPS = (50.773636, 5.154737)
DEFAULT_END_GPS = (50.773885, 5.154849)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rgb-bands",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=DEFAULT_RGB_BANDS,
        help="Band indices used to build the pseudo-RGB image (default: %(default)s)",
    )
    parser.add_argument(
        "--percentile-clip",
        type=float,
        default=98.0,
        help="Percentile used to clip intensities for visualization (default: %(default)s)",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=10,
        help="Histogram bins for the legacy RGB scaling (default: %(default)s)",
    )
    parser.add_argument(
        "--show-fig",
        action="store_true",
        help="Display the annotated map in a window after processing",
    )
    parser.add_argument(
        "--no-save-fig",
        action="store_true",
        help="Skip writing the annotated map to disk",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Dots-per-inch used for the exported annotated map (default: %(default)s)",
    )


def _add_runtime_args(parser: argparse.ArgumentParser, include_watch_folder: bool = True) -> None:
    if include_watch_folder:
        parser.add_argument(
            "--watch-folder",
            type=Path,
            default=Path("data"),
            help="Folder that will be scanned for new captures (default: %(default)s)",
        )
    parser.add_argument(
        "--processed-log",
        type=Path,
        default=Path("processed_folders.log"),
        help="Path to the log file that keeps track of processed folders",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/generated"),
        help="Directory where annotated maps will be stored",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=2048,
        help="Width of the sliding window in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--slide-step",
        type=int,
        default=320,
        help="Stride for the sliding window in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--tracker-match-threshold",
        type=int,
        default=75,
        help="Pixel distance threshold for associating detections across frames",
    )
    parser.add_argument(
        "--start-gps",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=DEFAULT_START_GPS,
        help="Start GPS coordinate of the scan (default: %(default)s)",
    )
    parser.add_argument(
        "--end-gps",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=DEFAULT_END_GPS,
        help="End GPS coordinate of the scan (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Confidence threshold forwarded to the YOLO detector (default: %(default)s)",
    )
    parser.add_argument(
        "--legacy-rgb",
        action="store_true",
        help="Use the legacy histogram-based RGB scaling technique",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lisa",
        description="Run the LISA hyperspectral vine monitoring pipeline",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously watch a folder for new captures and process them",
    )
    _add_runtime_args(watch_parser, include_watch_folder=True)
    _add_visualization_args(watch_parser)
    watch_parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Polling interval in seconds for checking new folders",
    )
    watch_parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process all pending folders once and exit",
    )

    process_parser = subparsers.add_parser(
        "process-folder",
        help="Process a specific capture folder",
    )
    process_parser.add_argument(
        "folder",
        type=Path,
        help="Path to the capture folder to process",
    )
    _add_runtime_args(process_parser, include_watch_folder=False)
    _add_visualization_args(process_parser)
    process_parser.add_argument(
        "--watch-folder",
        type=Path,
        default=None,
        help="Optional base folder used to resolve paths and logs (defaults to the folder's parent)",
    )
    process_parser.add_argument(
        "--update-log",
        action="store_true",
        help="Append the processed folder to the processed log file",
    )

    return parser.parse_args(argv)


def _build_runtime_config(args: argparse.Namespace, *, default_watch_folder: Path | None) -> PipelineRuntimeConfig:
    watch_folder = getattr(args, "watch_folder", None) or default_watch_folder or Path("data")

    visualization = VisualizationConfig(
        rgb_bands=tuple(int(v) for v in args.rgb_bands),
        percentile_clip=float(args.percentile_clip),
        histogram_bins=int(args.hist_bins),
        show_fig=bool(args.show_fig),
        save_fig=not bool(args.no_save_fig),
        dpi=int(args.plot_dpi),
    )

    return PipelineRuntimeConfig(
        watch_folder=watch_folder,
        processed_log_file=args.processed_log,
        poll_interval_seconds=getattr(args, "poll_interval", 10.0),
        slide_step=args.slide_step,
        window_width=args.window_width,
        start_point_gps=(float(args.start_gps[0]), float(args.start_gps[1])),
        end_point_gps=(float(args.end_gps[0]), float(args.end_gps[1])),
        tracker_match_threshold=args.tracker_match_threshold,
        results_dir=args.results_dir,
        visualization=visualization,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Loading models and cached datasets")
    assets = PipelineAssets.load()

    if args.command == "watch":
        runtime = _build_runtime_config(args, default_watch_folder=args.watch_folder)
        watch_for_new_folders(
            runtime,
            assets,
            use_legacy_rgb=args.legacy_rgb,
            confidence=args.confidence,
            run_once=args.run_once,
        )
        return 0

    if args.command == "process-folder":
        base_folder = args.watch_folder or args.folder.parent
        runtime = _build_runtime_config(args, default_watch_folder=base_folder)
        success = process_single_directory(
            args.folder,
            runtime,
            assets,
            use_legacy_rgb=args.legacy_rgb,
            confidence=args.confidence,
            update_log=args.update_log,
        )
        return 0 if success else 1

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
