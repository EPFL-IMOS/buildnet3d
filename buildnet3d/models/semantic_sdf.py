"""
Implementation of Semantic-SDF. This model is built on top of NeusFacto and adds
a 3D semantic segmentation model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, cast

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SemanticSDFModelConfig(NeuSModelConfig):
    """Semantic-SDF Model Config"""

    _target: Type = field(default_factory=lambda: SemanticSDFModel)
    semantic_loss_mult: float = 1.0
    """Factor that multiplies the semantic loss"""


class SemanticSDFModel(NeuSModel):
    """SemanticSDFModel extends NeuSFactoModel to add semantic segmentation in 3D."""

    config: SemanticSDFModelConfig

    def __init__(
        self, config: SemanticSDFModelConfig, metadata: Dict, **kwargs
    ) -> None:
        """
        To setup the model, provide a model `config` and the `metadata` from the
        outputs of the dataparser.
        """
        super().__init__(config=config, **kwargs)

        assert "semantics" in metadata.keys() and isinstance(
            metadata["semantics"], Semantics
        )
        self.colormap = metadata["semantics"].colors.clone().detach().to(self.device)

        self.color_mapping = {
            tuple(np.round(np.array(color), 3)): index
            for index, color in enumerate(metadata["semantics"].colors.tolist())
        }
        self._logger = logging.getLogger(__name__)

        self.step = 0

    def populate_modules(self) -> None:
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            use_average_appearance_embedding=(
                self.config.use_average_appearance_embedding
            ),
            spatial_distortion=self.scene_contraction,
        )

        self.renderer_semantics = SemanticRenderer()
        self.semantic_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Pass the `ray_bundle` through the model's field and renderer to get
        the model's output."""
        outputs = super().get_outputs(ray_bundle)

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor],
            samples_and_field_outputs["field_outputs"],
        )

        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=outputs["weights"]
        )

        # semantics colormaps
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Compute the loss dictionary from the `outputs` of the model, the `batch`
        that contains the ground truth data and the `metrics_dict`."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Semantic loss
        loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss(
            outputs["semantics"], batch["semantics"][..., 0].long().to(self.device)
        )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images from the `outputs` of the model and
        the `batch` which contains input and ground truth data."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # semantics
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        images_dict["semantics_colormap"] = self.colormap.to(self.device)[
            semantic_labels
        ]

        return metrics_dict, images_dict
