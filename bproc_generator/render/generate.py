import blenderproc as bproc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import tyro

sys.path.extend(["/home/chexu/buildnet3d"])
from buildnet3d.utils.utils import build_translation

@dataclass
class RenderParams:
    """Parameters for rendering synthetic building images."""
    
    load_scene: Path = Path("bproc_generator/data/example/House.obj")
    """Path to the segmented building OBJ file"""
    output_path: Path = Path("outputs/generated")
    """Output directory for rendered assets"""
    resolution: tuple[int, int] = (512, 512)
    """Rendering resolution (width, height)"""
    num_frames: int = 20
    """Number of camera frames to render"""
    enable_transparency: bool = False
    """Enable alpha channel in output images"""
    alpha: float = 1.0
    """Alpha value for transparent regions (0.0-1.0)"""
    
    # Camera parameters
    load_camera_path: Path = None
    """Optional path to predefined camera trajectory"""
    bound_thresh: int = 2
    """Pixel threshold for boundary object detection"""
    blank_thresh: int = 20
    """Pixel threshold for blank boundary detection"""
    retry_count: int = 30
    """Max attempts to find valid camera pose"""
    radius: float = 25
    """Camera orbit radius from point of interest"""
    theta_range: tuple[float, float] = (0.0, 2 * np.pi)
    """Azimuth angle range (radians)"""
    phi_range: tuple[float, float] = (0.0, 1.178)
    """Elevation angle range (radians)"""
    gamma_range: tuple[float, float] = (0.0, 0.0)
    """In-plane rotation range (radians)"""
    dist_thresh: float = 2.0
    """Minimum distance between camera poses to avoid overlap"""
    
    # Camera adjustment parameters
    delta_radius: tuple[float, float] = (0.5, 2.0)
    """Radius adjustment range during pose refinement"""
    delta_theta: tuple[float, float] = (0.05, 0.10)
    """Azimuth adjustment range during pose refinement"""
    delta_phi: tuple[float, float] = (0.05, 0.10)
    """Elevation adjustment range during pose refinement"""
    delta_poi: tuple[float, float] = (0.0, 0.5)
    """Point of interest adjustment range during pose refinement"""
    
    # Lighting parameters
    use_hdr_background: bool = True
    """Enable HDR environment lighting"""
    background_path: Path = Path("bproc_generator/data/example/zwartkops_straight_sunset_4k.hdr")
    """Path to HDR environment map"""
    hdr_strength: float = 1.0
    """Environment lighting intensity"""
    hdr_rotation: tuple[float, float, float] = (0.0, 0.0, 0.3926)
    """Environment rotation in radians (x,y,z)"""
    

@dataclass
class BlenderProcRenderer(RenderParams):
    """Generates synthetic building imagery with BlenderProc"""
    
    def __post_init__(self):     
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata: dict = {
            "camera_model" : "OPEN_CV",
            "width": self.resolution[0],
            "height": self.resolution[1],
            "has_mono_prior": False,
            "has_foreground_mask": False,
            "has_sparse_sfm_points": False,
            "scene_box" : {},
            "frames": [],
        }
        
        # Initialize rendering pipeline
        bproc.init()
        self.scene_objects = bproc.loader.load_obj(str(self.load_scene))
        self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.scene_objects) 

        self._assign_categories()
        self._setup_camera()
        self._setup_lighting()
        
        self.color_map = self._get_color_map()
        self.camera_idx = 0

    def _assign_categories(self):
        """Assigns semantic IDs to building components"""
        category_ids = [1, 3, 1, 1, 5, 4, 1, 2]  # Wall, Window, etc.
        for obj, cid in zip(self.scene_objects, category_ids):
            obj.set_cp("category_id", cid)
    
    def _setup_camera(self):
        """Initializes camera configuration"""
        self.camera_list: list[np.ndarray] = []
        bproc.camera.set_resolution(*self.resolution)
    
    def _setup_lighting(self):
        """Configures environment lighting"""
        if self.use_hdr_background:
            bproc.world.set_world_background_hdr_img(
                str(self.background_path),
                strength=self.hdr_strength,
                rotation_euler=self.hdr_rotation,
            )

    @staticmethod
    def _get_color_map() -> dict[str, list[float]]:
        """Provides semantic ID to RGB color mapping"""
        return {
            "0": [0, 0, 0],        # Background
            "1": [175, 200, 0],     # Wall
            "2": [0, 200, 200],     # Door
            "3": [125, 0, 200],     # Window
            "4": [175, 0, 200],     # Roof
            "5": [0, 50, 200],      # Chimney
            "6": [0, 200, 50],      # Vegetation
            "7": [150, 50, 50],     # Vehicle
            "8": [50, 175, 50],     # Furniture
            "9": [50, 50, 175],     # Misc
        }

    def _check_boundary(self, camera_pose: np.ndarray) -> tuple[list[bool], dict[str, list[bool]]]:
        """
        Validates object visibility within frame boundaries.
        Returns:
            tuple: [boundary_valid, content_valid] flags and per-edge detection results
        """
        bproc.camera.add_camera_pose(camera_pose, self.camera_idx)
        depth = bproc.camera.depth_via_raytracing(self.bvh_tree)

        # Edge analysis
        boundaries = {
            "top": [
                np.all(depth[:self.bound_thresh, :] == np.inf), 
                not np.all(depth[:self.blank_thresh, :] == np.inf),
            ],
            "bottom": [
                np.all(depth[-self.bound_thresh:, :] == np.inf),
                not np.all(depth[-self.blank_thresh:, :] == np.inf),
            ],
            "left": [
                np.all(depth[:, :self.bound_thresh] == np.inf),
                not np.all(depth[:, :self.blank_thresh] == np.inf),
            ],
            "right": [
                np.all(depth[:, -self.bound_thresh:] == np.inf),
                not np.all(depth[:, -self.blank_thresh:] == np.inf),
            ],
        }
        
        # Validation flags
        bound_valid = sum([v[0] for v in boundaries.values()]) == 4
        content_valid = sum([v[1] for v in boundaries.values()]) >= 3
        
        # Special case handling for horizontal/vertical coverage
        if boundaries["top"][1] and boundaries["bottom"][1]:
            content_valid = True
        if boundaries["left"][1] and boundaries["right"][1]:
            content_valid = True

        return [bound_valid, content_valid], boundaries

    def generate_camera_pose(self, point_of_interest: np.ndarray):
        """Generates valid camera pose through iterative refinement"""
        def random_camera_params():
            """Generates randomized spherical coordinates"""
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1])
            phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            # Perturb point of interest
            poi = point_of_interest + np.random.uniform(
                -self.delta_poi[1], self.delta_poi[1], size=3
            )
            return theta, phi, gamma, poi

        print(f" ----- Generating Camera Sample {self.camera_idx} ----- ")
        theta, phi, gamma, poi = random_camera_params()
        radius = self.radius
        retries = 0

        # Camera pose search loop
        while True:
            print(f"Retry {retries} for camera ID {self.camera_idx}", end="\r")
            location = build_translation(radius, theta, phi)
            rotation = bproc.camera.rotation_from_forward_vec(
                poi - location, inplane_rot=gamma
            )
            camera_pose = bproc.math.build_transformation_mat(location, rotation)

            # Boundary validation and adjustment
            (bound_valid, content_valid), boundaries = self._check_boundary(camera_pose)
            
            # Radius adjustment
            if not bound_valid:
                radius += np.random.uniform(*self.delta_radius)
            if not content_valid:
                radius -= np.random.uniform(*self.delta_radius)
                
            # Directional adjustments
            if not boundaries["top"][0] or not boundaries["bottom"][1]:
                poi[2] += np.random.uniform(*self.delta_poi)
                phi += np.random.uniform(*self.delta_phi)
            if not boundaries["bottom"][0] or not boundaries["top"][1]:
                poi[2] -= np.random.uniform(*self.delta_poi)
                phi -= np.random.uniform(*self.delta_phi)
            if not boundaries["left"][0] or not boundaries["right"][1]:
                theta -= np.random.uniform(*self.delta_theta)
            if not boundaries["right"][0] or not boundaries["left"][1]:
                theta += np.random.uniform(*self.delta_theta)
                
            # Pose acceptance criteria
            if bound_valid and content_valid:
                if all(np.linalg.norm(camera_pose[:3, 3] - p[:3, 3]) > self.dist_thresh
                       for p in self.camera_list):
                    self._add_camera_pose(camera_pose)
                    return
            
            retries += 1
            # Retry limit reached, reset parameters
            if retries >= self.retry_count:
                theta, phi, gamma, poi = random_camera_params()
                radius = self.radius
                retries = 0

    def _add_camera_pose(self, pose: np.ndarray):
        """Registers valid camera pose to pipeline"""
        bproc.camera.add_camera_pose(pose, self.camera_idx)
        self.camera_list.append(pose)
        self.metadata["frames"].append({
            "rgb_path": f"{self.camera_idx:04d}.png",
            "segmentation_path": f"{self.camera_idx:04d}_mask.png",
            "camera_to_world": pose.tolist(),
            "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
        })
        self.camera_idx += 1

    def save_images(self):
        """Converts HDF5 renders to standard image formats"""
        for subdir in ["images", "normals", "depths", "semantics", "instances"]:
            (self.output_path / subdir).mkdir(exist_ok=True)

        # Process each frame
        for i in range(self.camera_idx):
            with h5py.File(self.output_path / f"{i}.hdf5", "r") as f:
                # Process RGB
                rgb = np.array(f["colors"][:])
                if self.enable_transparency:
                    alpha = np.full((*rgb.shape[:2], 1), int(self.alpha * 255), dtype=np.uint8)
                    rgb = np.concatenate([rgb[..., :3], alpha], axis=-1)
                Image.fromarray(rgb).save(self.output_path / "images" / f"{i:04d}.png")
                
                # Process normals
                normal = (f["normals"][:] * 255).astype(np.uint8)
                Image.fromarray(normal).save(self.output_path / "normals" / f"{i:04d}_normal.png")
                
                # Process depth
                depth = (f["depth"][:] * 1000).astype(np.uint16)
                Image.fromarray(depth).save(self.output_path / "depths" / f"{i:04d}_depth.png")
                
                # Process semantics
                semantic = f["category_id_segmaps"][:]
                semantics = np.zeros((*semantic.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    semantics[semantic == int(j)] = color
                Image.fromarray(semantics).save(self.output_path / "semantics" / f"{i:04d}_mask.png")
                
                # Process instances
                instance = f["instance_segmaps"][:]
                instances = np.zeros((*instance.shape, 3), dtype=np.uint8)
                for j, color in self.color_map.items():
                    instances[instance == int(j)] = color
                Image.fromarray(instances).save(self.output_path / "instances" / f"{i:04d}.png")
            
            # Cleanup HDF5
            (self.output_path / f"{i}.hdf5").unlink()
            
    def save_metadata(self):
        """Saves camera metadata in JSON format"""
        with open(self.output_path / "meta_data.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

    def run(self):
        """Main rendering pipeline execution"""
        poi = bproc.object.compute_poi(self.scene_objects)
        for _ in range(self.num_frames):
            self.generate_camera_pose(poi)

        # Configure render outputs
        bproc.renderer.set_output_format(enable_transparency=self.enable_transparency)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])
        
        # Execute rendering and save results
        render_data = bproc.renderer.render()
        bproc.writer.write_hdf5(str(self.output_path), render_data)
        self.save_images()
        self.save_metadata()

def main():
    """Command-line entry point for rendering pipeline"""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(BlenderProcRenderer).run()
    
if __name__ == "__main__":
    main()