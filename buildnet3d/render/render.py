import json
import sys
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, field

import torch
import yaml
import imageio
import tyro

# Add custom paths for module imports
sys.path.extend([
    ".", 
    "/home/chexu/buildnet3d", 
    "/home/chexu/3D-WWR/semantic_sdf/"
])

from nerfstudio.scripts.render import BaseRender
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from buildnet3d.data.buildnet_dataset import BuildNetDataset

@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path using a trained pipeline."""

    load_camera_path: Path = Path("camera_path.json")
    output_format: Literal["images", "video"] = "images"
    rendered_resolution_scaling_factor: float = 0.5
    rendered_modalties: list[str] = field(default_factory=lambda: ["rgb", "semantics_colormap"])
    output_path: Path = Path("outputs/rendered")

    @staticmethod
    def load_trainer_config(
        config_path: Path,
        eval_num_rays_per_chunk: Optional[int] = None
    ) -> TrainerConfig:
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)

        config.pipeline.datamanager._target = VanillaDataManager[BuildNetDataset]
        if eval_num_rays_per_chunk:
            config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

        return config

    @staticmethod
    def load_path_from_json(camera_path: dict) -> Cameras:
        image_height = camera_path['height']
        image_width = camera_path['width']

        c2ws, fxs, fys = [], [], []
        for frame in camera_path['frames']:
            c2w = torch.tensor(frame['camera_to_world']).view(4, 4)[:3]
            intrinsics = torch.tensor(frame['intrinsics']).view(3, 3)
            c2ws.append(c2w)
            fxs.append(intrinsics[0, 0])
            fys.append(intrinsics[1, 1])

        return Cameras(
            fx=torch.tensor(fxs),
            fy=torch.tensor(fys),
            cx=image_width / 2,
            cy=image_height / 2,
            camera_to_worlds=torch.stack(c2ws, dim=0),
            camera_type=CameraType.PERSPECTIVE,
            times=None,
        )

    def load_pipeline(self, test_mode: Literal["test", "val", "inference"] = "inference") -> Pipeline:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = self.config.pipeline.setup(device=device, test_mode=test_mode)
        assert isinstance(pipeline, Pipeline)
        pipeline.eval()

        # Load the checkpoint
        checkpoint_dir = self.config.get_checkpoint_dir()
        self.config.load_dir = checkpoint_dir
        steps = [int(p.name.split("-")[1].split(".")[0]) for p in checkpoint_dir.glob("step-*.ckpt")]
        self.config.load_step = max(steps)
        ckpt_path = checkpoint_dir / f"step-{self.config.load_step:09d}.ckpt"
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found."

        loaded_state = torch.load(ckpt_path, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

        return pipeline

    def main(self) -> None:
        self.config = self.load_trainer_config(config_path=self.load_config)
        self.pipeline = self.load_pipeline()

        with self.load_camera_path.open("r") as f:
            camera_dict = json.load(f)

        cameras = self.load_path_from_json(camera_dict)
        cameras.rescale_output_resolution(self.rendered_resolution_scaling_factor)
        cameras = cameras.to(self.pipeline.device)

        num_frames = cameras.size

        # Prepare directories and buffers
        output_root = self.output_path
        rendered_images = {modality: [] for modality in self.rendered_modalties}

        for idx in range(num_frames):
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(cameras[idx : idx + 1])
                for modality in self.rendered_modalties:
                    if modality in outputs:
                        img = outputs[modality].cpu().numpy()
                        img = (img * 255).clip(0, 255).astype("uint8")

                        rendered_images[modality].append(img)
        
        if self.output_format == "images":
            for modality, images in rendered_images.items():
                output_dir = output_root / modality
                output_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    imageio.imwrite(output_dir / f"frame_{i:04d}.png", img)
        elif self.output_format == "video":
            for modality, images in rendered_images.items():
                output_dir = output_root / modality
                output_dir.mkdir(parents=True, exist_ok=True)
                video_path = output_dir / "rendered_video.mp4"
                imageio.mimwrite(video_path, images, fps=20)
        
def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderCameraPath).main()

if __name__ == "__main__":
    entrypoint()