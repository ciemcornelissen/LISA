from __future__ import annotations

"""Utilities for loading and reusing heavy pipeline assets."""

from dataclasses import dataclass
from typing import Any, Dict

from lisa.inference import load_models_data


@dataclass(slots=True)
class PipelineAssets:
    """Container for the heavy components that are reused across folders."""

    models: Dict[str, Any]
    weight_config_params: Dict[str, Any]
    weight_images: Any
    images: Any
    config: Dict[str, Any]

    @property
    def weight_config(self) -> Dict[str, Any]:
        """Shortcut to the nested weight configuration."""

        return self.weight_config_params["config"]

    @classmethod
    def load(cls) -> "PipelineAssets":
        """Load all models, weight parameters, and cached datasets."""

        models, weight_config_params, weight_images, images, config = load_models_data()
        # Downstream logic expects the dataset object to carry a normalize flag.
        if hasattr(images, "normalize"):
            images.normalize = 1
        return cls(models, weight_config_params, weight_images, images, config)