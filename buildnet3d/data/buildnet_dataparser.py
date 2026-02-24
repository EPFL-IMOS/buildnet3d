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
    """Whether or not to load monocular depth and normal."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters (default: mm -> m)."""
    include_foreground_mask: bool = False
    """Whether or not to load foreground mask."""
    downscale_factor: int = 1
    """Downscale image size."""
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the axis-aligned bbox will be scaled to this value.
    """
    split_by_filename: bool = True
    """If True, use filename tags (frame_train_ / frame_eval_) for splitting."""
    skip_every_for_val_split: int = 1
    """Used when split_by_filename=False. Every Nth frame goes to eval."""
    auto_orient: bool = True
    """Orient poses correctly."""
    semantics: bool = False
    """Whether or not to load semantics."""


@dataclass
class BuildNet(DataParser):
    """BuildNet Dataset"""

    config: BuildNetDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_root = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "meta_data.json")
            data_root = self.config.data

        use_mono_prior: bool = ("has_mono_prior" in meta and meta["has_mono_prior"] is True)

        image_filenames = []
        segmentation_filenames = []
        depth_filenames = []
        normal_filenames = []
        rgb_names = []

        transform = None
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for frame in meta["frames"]:
            rgb_name = Path(frame["rgb_path"]).name
            rgb_names.append(rgb_name)

            image_filenames.append(data_root / "images" / frame["rgb_path"])

            if self.config.semantics:
                segmentation_filenames.append(data_root / "semantic" / frame["segmentation_path"])

            if use_mono_prior:
                depth_filenames.append(data_root / frame["depth_path"])
                normal_filenames.append(data_root / frame["normals_path"])

            intrinsics = torch.tensor(frame["intrinsics"], dtype=torch.float32)
            camtoworld = torch.tensor(frame["camtoworld"], dtype=torch.float32)

            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        # Stack tensors
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        c2w_colmap_full = torch.stack(camera_to_worlds)

        # -------- Data Split (filename OR interval) --------
        all_indices = list(range(len(image_filenames)))
        train_indices = []
        eval_indices = []

        if self.config.split_by_filename:
            for i, rgb_name in enumerate(rgb_names):
                if "train" in rgb_name:
                    train_indices.append(i)
                elif "eval" in rgb_name:
                    eval_indices.append(i)
                else:
                    raise ValueError(
                        f"Unrecognized rgb filename '{rgb_name}'. "
                        "Expected names containing 'frame_train_' or 'frame_eval_'."
                    )
        else:
            n = self.config.skip_every_for_val_split
            if n <= 1:
                # fallback: no real split
                train_indices = all_indices
                eval_indices = all_indices
                CONSOLE.log(
                    "[BuildNet] skip_every_for_val_split <= 1, "
                    "using all images for both train and eval."
                )
            else:
                eval_indices = all_indices[::n]
                eval_set = set(eval_indices)
                train_indices = [i for i in all_indices if i not in eval_set]

        if split == "train":
            indices = train_indices
        elif split in ["val", "test", "eval"]:
            indices = eval_indices
        else:
            raise ValueError(f"Unknown split '{split}'")

        if len(indices) == 0:
            if self.config.split_by_filename:
                raise ValueError(
                    f"No images found for split='{split}'. "
                    "Check rgb_path naming (expected frame_train_* or frame_eval_*)."
                )
            raise ValueError(
                f"No images found for split='{split}'. "
                f"Check skip_every_for_val_split={self.config.skip_every_for_val_split}."
            )

        # Apply split to file lists
        image_filenames = [image_filenames[i] for i in indices]

        if self.config.semantics:
            segmentation_filenames = [segmentation_filenames[i] for i in indices]

        if use_mono_prior:
            depth_filenames = [depth_filenames[i] for i in indices]
            normal_filenames = [normal_filenames[i] for i in indices]

        # Apply split to camera tensors
        fx = fx[indices]
        fy = fy[indices]
        cx = cx[indices]
        cy = cy[indices]
        c2w_colmap = c2w_colmap_full[indices].clone()
        camera_to_worlds = c2w_colmap.clone()

        # Auto-orient on the selected split
        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="none",
                center_method="focus",
            )

        # Scene box
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(aabb=aabb)

        # Cameras
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

        # Segmentation information
        semantics_obj = None
        if self.config.semantics:
            panoptic_classes = load_from_json(data_root / "segmentation_data.json")
            classes = list(panoptic_classes.keys())
            colors = torch.tensor(list(panoptic_classes.values()), dtype=torch.float32)
            semantics_obj = Semantics(
                filenames=segmentation_filenames,
                classes=classes,
                colors=colors,
                mask_classes=[],
            )

        # Helpful log
        split_mode = "filename" if self.config.split_by_filename else "interval"
        CONSOLE.log(
            f"[BuildNet] mode={split_mode}, split='{split}': {len(indices)} images "
            f"(train={len(train_indices)}, eval={len(eval_indices)})"
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "transform": transform,
                "semantics": semantics_obj,
                "camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None,
                "include_mono_prior": use_mono_prior,
                "depth_filenames": depth_filenames if use_mono_prior else None,
                "normal_filenames": normal_filenames if use_mono_prior else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs