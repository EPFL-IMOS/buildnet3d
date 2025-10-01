from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class BuildNetDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: BuildNet)
    """Target class to instantiate"""
    data: Path = Path("data/buildnet")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter
    to meter conversion."""
    include_foreground_mask: bool = False
    """Whether or not to load foreground mask"""
    downscale_factor: int = 1
    """Downscale image size"""
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the axis-aligned bbox will be scaled to this value.
    """
    skip_every_for_val_split: int = 1
    """Sub sampling validation images"""
    auto_orient: bool = True
    """Orient poses corectly"""


@dataclass
class BuildNet(DataParser):
    """BuildNet Dataset"""

    config: BuildNetDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
        else:
            meta = load_from_json(self.config.data / "meta_data.json")

        use_mono_prior: bool = (
            "has_mono_prior" in meta.keys() and meta["has_mono_prior"] is True
        )

        image_filenames = []
        segmentation_filenames = []
        depth_filenames = []
        normal_filenames = []
        transform = None
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for frame in meta["frames"]:
            image_filenames.append(self.config.data / "images"  / frame["rgb_path"])
            segmentation_filenames.append(self.config.data / "semantic" / frame["segmentation_path"])
            # sensor_depth_filenames.append(self.config.data / "depths"  / frame["rgb_path"].replace(".png", "_depth.png"))
            if use_mono_prior:
                depth_filenames.append(self.config.data / frame["depth_path"])
                normal_filenames.append(self.config.data / frame["normals_path"])

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        c2w_colmap = torch.stack(camera_to_worlds)
        camera_to_worlds = torch.stack(camera_to_worlds)

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="none",
                center_method="focus",
            )

        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
        )

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # segmentation information
        panoptic_classes = load_from_json(self.config.data / "segmentation_data.json")
        classes = list(panoptic_classes.keys())
        colors = (
            torch.tensor(list(panoptic_classes.values()), dtype=torch.float32)
        )
        semantics = Semantics(
            filenames=segmentation_filenames,
            classes=classes,
            colors=colors,
            mask_classes=[]
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "transform": transform,
                "semantics": semantics,
                "camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None,
                "include_mono_prior": use_mono_prior,
                "depth_filenames": depth_filenames if use_mono_prior else None,
                "normal_filenames": normal_filenames if use_mono_prior else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs
