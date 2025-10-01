"""
Field for semantic-SDF, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with
extracting high fidelity surfaces, in addition with a 3D semantic segmentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames, SemanticFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from torch import Tensor, nn


@dataclass
class SemanticSDFFieldConfig(SDFFieldConfig):
    """Semantic-SDF Field Config"""

    _target: Type = field(default_factory=lambda: SemanticSDFField)
    num_semantic_classes: int = 6
    """Number of classes in the segmentation"""
    semantic_mlp_number_layers: int = 2
    """Number of layers for the semantic neural network"""
    semantic_mlp_layer_width: int = 128
    """Numbers of neurons in one layer of the semantic MLP"""


class SemanticSDFField(SDFField):
    """
    A field that learns a Signed Distance Functions (SDF), an RGB color and a
    semantic segmentation.
    """

    config: SemanticSDFFieldConfig

    def __init__(
        self,
        config: SemanticSDFFieldConfig,
        aabb: Float[Tensor, "2 3"],  # noqa: F722
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        """
        The field is built on top of the SDF field. It needs a `config` file, the scene
        size as an axis-aligned bounding box in `aabb` and the number of images used for
        embedding appearance in `num_images`. Set `use_average_appearance_embedding` if
        necessary and specify the `spatial_distortion` if there is some.
        """
        super().__init__(
            config,
            aabb,
            num_images,
            use_average_appearance_embedding,
            spatial_distortion,
        )

        self.num_semantic_classes = self.config.num_semantic_classes

        self.mlp_semantic = MLP(
            in_dim=self.config.geo_feat_dim,
            layer_width=self.config.semantic_mlp_layer_width,
            num_layers=self.config.semantic_mlp_number_layers,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(),
            num_classes=self.num_semantic_classes,
        )

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        Compute output of the field using the `ray_samples` as input.
        `density_embedding` is a useless artifact from base class. Use
        `return_aphas` if needed
        """
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(
                hidden_output, [1, self.config.geo_feat_dim], dim=-1
            )
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        rgb = self.get_colors(
            inputs, directions_flat, gradients, geo_feature, camera_indices
        )

        semantics_input = geo_feature.view(-1, self.config.geo_feat_dim)
        semantics_output = self.mlp_semantic(semantics_input)
        semantics = self.field_head_semantic(semantics_output)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)
        semantics = semantics.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
                FieldHeadNames.SEMANTICS: semantics,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs
