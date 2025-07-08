import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.extend(["/home/chexu/buildnet3d"])

import matplotlib.pyplot as plt
import numpy as np

from buildnet3d.utils.utils import (
    build_rotation_from_forward_vec,
    build_transformation,
    build_translation,
)


@dataclass
class CreateCameraPath:
    """Generate a camera path on a sphere around a point of interest (POI)."""

    # Camera Path
    radius: float = 1.0
    start_angle: List[float] = (np.deg2rad(-75), np.deg2rad(35))  # [theta, phi]
    end_angle: List[float] = (np.deg2rad(45), np.deg2rad(-10))  # [theta, phi]
    gamma: float = 0.0  # In-plane rotation
    point_of_interest: np.ndarray = np.array([0, 0, 0])  # Look-at point
    num_steps: int = 10  # Camera steps

    # Camera Intrinsics
    image_height: float = 512.0
    image_width: float = 512.0
    focal_length_x: float = 450.0
    focal_length_y: float = 450.0

    # Output
    output_path: Path = Path("outputs/")

    def interpolate_camera_path(self) -> List[np.ndarray]:
        """Interpolate camera trajectory from start to end angles."""
        theta1, phi1 = self.start_angle
        theta2, phi2 = self.end_angle

        trajectory = []
        for i in range(self.num_steps):
            alpha = i / (self.num_steps - 1)
            theta = (1 - alpha) * theta1 + alpha * theta2
            phi = (1 - alpha) * phi1 + alpha * phi2

            # Convert spherical coordinates to Cartesian coordinates
            position = build_translation(self.radius, theta, phi)

            # rotate the camera to look at the point of interest
            forward_vec = self.point_of_interest - position
            rotation = build_rotation_from_forward_vec(
                forward_vec, inplane_rot=self.gamma
            )

            # Build the transformation matrix
            transformation = build_transformation(position, rotation)

            trajectory.append(transformation)

        return trajectory

    def save_camera_path(self, trajectory: List[np.ndarray]) -> None:
        camera_path = {
            "camera_type": "perspective",
            "height": self.image_height,
            "width": self.image_width,
            "frames": [],
        }
        camera_intrinsic = np.array(
            [
                [self.focal_length_x, 0, self.image_width / 2],
                [0, self.focal_length_y, self.image_height / 2],
                [0, 0, 1],
            ]
        ).tolist()

        for pose in trajectory:
            camera_to_world = pose.tolist()
            camera_path["frames"].append(
                {"camera_to_world": camera_to_world, "intrinsics": camera_intrinsic}
            )

        # Save the camera path to a JSON file
        output_file = self.output_path / "cameras.json"
        print(f"Saving camera path to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(camera_path, f, indent=4)

    def visualize_camera_path(self, trajectory: List[np.ndarray]) -> None:
        """Visualize the trajectory on a 3D sphere."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the sphere
        u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
        x = self.radius * np.cos(u) * np.sin(v)
        y = self.radius * np.sin(u) * np.sin(v)
        z = self.radius * np.cos(v)
        ax.plot_surface(x, y, z, color="lightgray", alpha=0.2)

        # Plot trajectory
        traj = np.array([pose[:3, 3] for pose in trajectory])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "r-", label="Camera Path")
        ax.scatter(*traj[0], color="blue", label="Start")
        ax.scatter(*traj[-1], color="green", label="End")

        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.savefig(self.output_path / "camera_trajectory.png")

    def run(self) -> None:
        poses = self.interpolate_camera_path()
        self.visualize_camera_path(poses)
        self.save_camera_path(poses)


if __name__ == "__main__":
    CreateCameraPath().run()
