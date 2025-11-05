from __future__ import annotations

"""Visualization helpers for annotated vineyard maps."""

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def create_annotated_map(
    image_input: Image.Image | np.ndarray | str,
    results_list: Iterable[Mapping[str, object]],
    output_path: str | Path = "results/final_annotated_map.png",
    *,
    save_fig: bool = True,
    show_fig: bool = False,
    dpi: int = 300,
) -> Path:
    """Overlay bunch predictions on a pseudo-RGB image and save the figure."""

    if isinstance(image_input, str | Path):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        array = image_input
        if array.dtype != np.uint8:
            if array.max() <= 1.0 and array.min() >= 0.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = ((array - array.min()) / (array.ptp() or 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(array)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)!r}")

    img_width, img_height = image.size
    fig_width_in = img_width / dpi
    fig_height_in = img_height / dpi

    fig, ax = plt.subplots(1, figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.imshow(image)

    line_width = max(int(img_width / 4000), 1)
    font_size = max(int(img_width / 400), 8)
    text_offset = max(int(img_width / 4000), 2)

    for result in results_list:
        bbox = result.get("global_bounding_box")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=line_width,
                edgecolor="lime",
                facecolor="none",
            )
        )

        label = (
            f"ID:{result.get('bunch_id', 'N/A')} | "
            f"Brix:{result.get('predicted_brix', 0):.1f} | "
            f"Acid:{result.get('predicted_acidity', 0):.1f} | "
            f"W:{result.get('predicted_weight_g', 0):.0f}g"
        )
        ax.text(
            x1,
            y1 - text_offset,
            label,
            color="white",
            fontsize=font_size,
            bbox={
                "facecolor": "black",
                "alpha": 0.7,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.3",
            },
        )

    ax.axis("off")
    plt.tight_layout(pad=0)

    output_path = Path(output_path)
    if save_fig:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    if show_fig:
        plt.show()
    plt.close(fig)
    return output_path


__all__ = ["create_annotated_map"]
