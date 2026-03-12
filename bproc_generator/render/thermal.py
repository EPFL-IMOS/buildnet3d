import blenderproc as bproc

import json
import sys
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Optional

import bmesh
import bpy
import h5py
import matplotlib
import numpy as np
import tyro
from PIL import Image
from scipy.spatial import cKDTree

sys.path.extend(["/home/chexu/buildnet3d"])
from blenderproc.python.types.MeshObjectUtility import convert_to_meshes
from buildnet3d.utils.utils import build_translation, load_temperatures_from_csv


@dataclass
class RenderParams:
    """Parameters for rendering STL geometry and thermal images from a temperature point cloud."""

    load_scene: Path = Path("bproc_generator/data/example/bunny.stl")
    """Path to STL mesh"""
    output_path: Path = Path("outputs/generated")
    """Output directory for rendered assets"""
    resolution: tuple[int, int] = (512, 512)
    """Rendering resolution (width, height)"""
    num_frames: int = 1
    """Number of camera frames to render"""
    enable_transparency: bool = False
    """Enable alpha channel in RGB output images"""
    alpha: float = 1.0
    """Alpha value for transparent regions (0.0-1.0)"""

    # Thermal rendering
    thermal_csv: Optional[Path] = Path("bproc_generator/data/example/bunny_heat_surface_steel.csv")
    """Path to point-cloud temperature CSV with columns x,y,z,temperature"""
    render_thermal: bool = True
    """Render thermal image from point cloud"""
    thermal_point_size: int = 2
    """Projected thermal point splat size in pixels"""
    thermal_cmap: str = "inferno"
    """Matplotlib colormap for thermal visualization"""
    thermal_min: Optional[float] = None
    """Optional fixed minimum temperature"""
    thermal_max: Optional[float] = None
    """Optional fixed maximum temperature"""
    thermal_depth_epsilon: float = 0.01
    """Visibility tolerance when comparing thermal depth against mesh depth"""
    thermal_background: tuple[int, int, int] = (0, 0, 0)
    """RGB background color for thermal images"""
    thermal_k_neighbors: int = 2
    """Number of nearest thermal points used for mesh vertex temperature interpolation"""

    # Coordinate convention for thermal CSV
    thermal_auto_center_to_mesh: bool = False
    """Translate thermal cloud center to STL mesh center"""
    thermal_auto_scale_to_mesh: bool = False
    """Scale thermal cloud bbox to STL mesh bbox"""

    # Camera parameters
    load_camera_path: Optional[Path] = Path("bproc_generator/data/example/cameras.json")
    """Optional path to predefined camera trajectory"""
    bound_thresh: int = 2
    """Pixel threshold for boundary object detection"""
    blank_thresh: int = 75
    """Pixel threshold for blank boundary detection"""
    retry_count: int = 30
    """Max attempts to find valid camera pose"""
    radius: float = 0.25
    """Camera orbit radius from point of interest"""
    theta_range: tuple[float, float] = (0.0, 2 * np.pi)
    """Azimuth angle range (radians)"""
    phi_range: tuple[float, float] = (0.0, 1.178)
    """Elevation angle range (radians)"""
    gamma_range: tuple[float, float] = (0.0, 0.0)
    """In-plane rotation range (radians)"""
    dist_thresh: float = 0.005
    """Minimum distance between camera poses to avoid overlap"""

    # Camera adjustment
    delta_radius: tuple[float, float] = (0.001, 0.005)
    """Radius adjustment range during pose refinement"""
    delta_theta: tuple[float, float] = (0.05, 0.10)
    """Azimuth adjustment range during pose refinement"""
    delta_phi: tuple[float, float] = (0.05, 0.10)
    """Elevation adjustment range during pose refinement"""
    delta_poi: tuple[float, float] = (0.0, 0.001)
    """Point of interest adjustment range during pose refinement"""

    # Lighting
    use_hdr_background: bool = False
    background_path: Path = Path("bproc_generator/data/example/zwartkops_straight_sunset_4k.hdr")
    hdr_strength: float = 1.0
    hdr_rotation: tuple[float, float, float] = (0.0, 0.0, 0.3926)

@dataclass
class ThermalViewCache:
    """Geometry-dependent cache for one fixed view."""

    point_px: Optional[np.ndarray] = None
    point_py: Optional[np.ndarray] = None
    point_depth: Optional[np.ndarray] = None
    point_linear_idx: Optional[np.ndarray] = None

    mesh_px: Optional[np.ndarray] = None
    mesh_py: Optional[np.ndarray] = None
    mesh_vertex_ids: Optional[np.ndarray] = None
    mesh_bary: Optional[np.ndarray] = None


@dataclass
class BlenderProcRenderer(RenderParams):
    """Renders STL geometry with BlenderProc and dense thermal images from a temperature point cloud."""

    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata: dict = {
            "camera_model": "OPEN_CV",
            "width": self.resolution[0],
            "height": self.resolution[1],
            "has_mono_prior": False,
            "has_foreground_mask": False,
            "has_sparse_sfm_points": False,
            "scene_box": {"aabb": [[], []]},
            "frames": [],
        }

        bproc.init()

        self.scene_objects = self._load_stl_as_mesh_objects(self.load_scene)
        if len(self.scene_objects) == 0:
            raise RuntimeError(f"No mesh objects were loaded from: {self.load_scene}")
        self.bvh_tree = bproc.object.create_bvh_tree_multi_objects(self.scene_objects)

        self._assign_categories()
        self._setup_camera()
        self._setup_lighting()

        self.color_map = self._get_color_map()
        self.camera_idx = 0

        mesh_min, mesh_max = self._compute_scene_bbox()
        mesh_size = mesh_max - mesh_min
        self.metadata["scene_box"]["aabb"][0] = (mesh_min - 0.25 * mesh_size).tolist()
        self.metadata["scene_box"]["aabb"][1] = (mesh_max + 0.25 * mesh_size).tolist()

        # Thermal point cloud data
        self.thermal_points_world: Optional[np.ndarray] = None
        self.thermal_values: Optional[np.ndarray] = None
        self.thermal_timestamps: Optional[np.ndarray] = None
        self.thermal_node_ids: Optional[np.ndarray] = None

        # Geometry for thermal rendering
        self.mesh_vertices_world: Optional[np.ndarray] = None
        self.mesh_faces: Optional[np.ndarray] = None
        self.mesh_interp_indices: Optional[np.ndarray] = None
        self.mesh_interp_weights: Optional[np.ndarray] = None

        self.thermal_render_cache: dict[int, ThermalViewCache] = {}

        if self.render_thermal and self.thermal_csv is not None:
            self._load_thermal_cloud()
            self.mesh_vertices_world, self.mesh_faces = self._extract_scene_mesh_world()
            self.mesh_interp_indices, self.mesh_interp_weights = self._build_mesh_temperature_interpolator()

    def _load_stl_as_mesh_objects(self, stl_path: Path):
        """Import STL with bpy, then wrap imported Blender objects as BlenderProc MeshObjects."""
        before_names = set(obj.name for obj in bpy.context.scene.objects)

        bpy.ops.wm.stl_import(
            filepath=str(stl_path),
            global_scale=1.0,
            use_scene_unit=False,
            use_facet_normal=True,
            forward_axis="Y",
            up_axis="Z",
        )

        after_objects = list(bpy.context.scene.objects)
        imported = [obj for obj in after_objects if obj.name not in before_names and obj.type == "MESH"]
        if len(imported) == 0:
            raise RuntimeError(f"Failed to import STL: {stl_path}")

        return convert_to_meshes(imported)

    def _assign_categories(self):
        """Assign semantic IDs to imported objects."""
        for obj in self.scene_objects:
            obj.set_cp("category_id", 1)

    def _setup_camera(self):
        self.camera_list: list[np.ndarray] = []
        bproc.camera.set_resolution(*self.resolution)

    def _setup_lighting(self):
        if self.use_hdr_background:
            bproc.world.set_world_background_hdr_img(
                str(self.background_path),
                strength=self.hdr_strength,
                rotation_euler=self.hdr_rotation,
            )

    @staticmethod
    def _get_color_map() -> dict[str, list[float]]:
        return {
            "0": [0, 0, 0],
            "1": [175, 200, 0],
            "2": [0, 200, 200],
            "3": [125, 0, 200],
            "4": [175, 0, 200],
            "5": [0, 50, 200],
            "6": [0, 200, 50],
            "7": [150, 50, 50],
            "8": [50, 175, 50],
            "9": [50, 50, 175],
        }

    def _compute_scene_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        mins = []
        maxs = []
        for obj in self.scene_objects:
            bb = np.array(obj.get_bound_box(), dtype=np.float64)
            mins.append(bb.min(axis=0))
            maxs.append(bb.max(axis=0))
        return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)

    def _load_thermal_cloud(self):
        points, temps, timestamps, node_ids = load_temperatures_from_csv(self.thermal_csv)

        if self.thermal_auto_center_to_mesh or self.thermal_auto_scale_to_mesh:
            mesh_min, mesh_max = self._compute_scene_bbox()
            pc_min = points.min(axis=0)
            pc_max = points.max(axis=0)

            pc_center = 0.5 * (pc_min + pc_max)
            mesh_center = 0.5 * (mesh_min + mesh_max)

            points = points - pc_center

            if self.thermal_auto_scale_to_mesh:
                pc_extent = np.maximum(pc_max - pc_min, 1e-8)
                mesh_extent = np.maximum(mesh_max - mesh_min, 1e-8)
                scale = float(np.min(mesh_extent / pc_extent))
                points = points * scale

            if self.thermal_auto_center_to_mesh:
                points = points + mesh_center
            else:
                points = points + pc_center

        self.thermal_points_world = points
        self.thermal_values = temps
        self.thermal_timestamps = timestamps
        self.thermal_node_ids = node_ids

        self.thermal_min = self.thermal_min if self.thermal_min is not None else float(temps.min())
        self.thermal_max = self.thermal_max if self.thermal_max is not None else float(temps.max())
        self.metadata["temperature_range"] = [self.thermal_min, self.thermal_max]

    def _extract_scene_mesh_world(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract triangulated mesh vertices/faces from imported STL objects in world coordinates.
        Returns:
            vertices_world: (N, 3)
            faces:          (M, 3)
        """
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for obj in self.scene_objects:
            blender_obj = obj.blender_obj
            mesh = blender_obj.data

            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

            verts_local = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
            faces_local = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

            M = np.array(blender_obj.matrix_world, dtype=np.float64)
            verts_h = np.concatenate([verts_local, np.ones((len(verts_local), 1))], axis=1)
            verts_world = (M @ verts_h.T).T[:, :3]

            all_vertices.append(verts_world)
            all_faces.append(faces_local + vertex_offset)
            vertex_offset += len(verts_world)

        vertices_world = np.concatenate(all_vertices, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        return vertices_world, faces

    def _build_mesh_temperature_interpolator(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute inverse-distance interpolation from thermal points to mesh vertices.
        Returns:
            idxs:    (N_vertices, k) nearest thermal point indices
            weights: (N_vertices, k) normalized interpolation weights
        """
        if self.mesh_vertices_world is None or self.thermal_points_world is None:
            raise RuntimeError("Mesh vertices or thermal points are not available.")

        k = max(1, self.thermal_k_neighbors)
        tree = cKDTree(self.thermal_points_world)
        dists, idxs = tree.query(self.mesh_vertices_world, k=k)

        if k == 1:
            idxs = idxs[:, None]
            weights = np.ones((len(self.mesh_vertices_world), 1), dtype=np.float64)
            return idxs, weights

        dists = np.maximum(dists, 1e-8)
        weights = 1.0 / dists
        weights /= np.sum(weights, axis=1, keepdims=True)
        return idxs, weights

    def _compute_mesh_vertex_temperatures(self, point_temperatures: np.ndarray) -> np.ndarray:
        """Interpolate one thermal frame from points to mesh vertices."""
        if self.mesh_interp_indices is None or self.mesh_interp_weights is None:
            raise RuntimeError("Mesh interpolation cache is not initialized.")

        return np.sum(
            self.mesh_interp_weights * point_temperatures[self.mesh_interp_indices],
            axis=1,
        )

    def _check_boundary(self, camera_pose: np.ndarray) -> tuple[list[bool], dict[str, list[bool]]]:
        # NOTE:
        # BlenderProc does not expose a super-clean "probe pose without registering" API here.
        # This keeps the current behavior, but uses the same index each retry.
        # If BlenderProc appends instead of overwriting, this is the one place to verify experimentally.
        bproc.camera.add_camera_pose(camera_pose, self.camera_idx)
        depth = bproc.camera.depth_via_raytracing(self.bvh_tree)

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

        bound_valid = sum(v[0] for v in boundaries.values()) == 4
        content_valid = sum(v[1] for v in boundaries.values()) >= 3

        if boundaries["top"][1] and boundaries["bottom"][1]:
            content_valid = True
        if boundaries["left"][1] and boundaries["right"][1]:
            content_valid = True

        return [bound_valid, content_valid], boundaries

    def generate_camera_pose(self, point_of_interest: np.ndarray):
        def random_camera_params():
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1])
            phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            poi = point_of_interest + np.random.uniform(-self.delta_poi[1], self.delta_poi[1], size=3)
            return theta, phi, gamma, poi

        print(f" ----- Generating Camera Sample {self.camera_idx} ----- ")
        theta, phi, gamma, poi = random_camera_params()
        radius = self.radius
        retries = 0

        while True:
            print(f"Retry {retries} for camera ID {self.camera_idx}", end="\r")
            location = build_translation(radius, theta, phi)
            rotation = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=gamma)
            camera_pose = bproc.math.build_transformation_mat(location, rotation)

            (bound_valid, content_valid), boundaries = self._check_boundary(camera_pose)

            if not bound_valid:
                radius += np.random.uniform(*self.delta_radius)
            if not content_valid:
                radius -= np.random.uniform(*self.delta_radius)

            print(f"Boundaries: {boundaries}, Valid: {bound_valid}, Content Valid: {content_valid}", end="\r")

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

            if bound_valid and content_valid:
                if all(np.linalg.norm(camera_pose[:3, 3] - p[:3, 3]) > self.dist_thresh for p in self.camera_list):
                    self._add_camera_pose(camera_pose)
                    return

            retries += 1
            if retries >= self.retry_count:
                theta, phi, gamma, poi = random_camera_params()
                radius = self.radius
                retries = 0

    def _add_camera_pose(self, pose: np.ndarray):
        bproc.camera.add_camera_pose(pose, self.camera_idx)
        self.camera_list.append(pose)
        self.metadata["frames"].append({
            "rgb_path": f"images/{self.camera_idx:04d}.png",
            "segmentation_path": f"semantics/{self.camera_idx:04d}_mask.png",
            "depth_path": f"depths/{self.camera_idx:04d}_depth.png",
            "thermal_path": f"thermal/{self.camera_idx:04d}_thermal.png" if self.render_thermal else None,
            "camera_to_world": pose.tolist(),
            "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
        })
        self.camera_idx += 1

    @staticmethod
    def _normalize_temperature(values: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
        if vmin is None:
            vmin = float(np.min(values))
        if vmax is None:
            vmax = float(np.max(values))
        if vmax <= vmin:
            raise ValueError("thermal_max must be greater than thermal_min.")
        return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    def _temperature_to_rgb(self, values: np.ndarray) -> np.ndarray:
        normalized = self._normalize_temperature(values, self.thermal_min, self.thermal_max)
        cmap = matplotlib.colormaps[self.thermal_cmap]
        return (cmap(normalized)[:, :3] * 255.0).astype(np.uint8)

    @staticmethod
    def _world_to_camera(points_world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
        world2cam = np.linalg.inv(cam2world)
        points_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
        points_cam = (world2cam @ points_h.T).T[:, :3]
        return points_cam

    @staticmethod
    def _barycentric_coords_2d(
        px: float,
        py: float,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute barycentric coordinates of point p with respect to 2D triangle abc."""
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-12:
            return -1.0, -1.0, -1.0

        w1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
        w2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
        w3 = 1.0 - w1 - w2
        return w1, w2, w3

    def _build_thermal_render_cache(
        self,
        K: np.ndarray,
        cam2world: np.ndarray,
        depth_map: np.ndarray,
    ) -> ThermalViewCache:
        """
        Precompute all geometry-dependent quantities for a fixed camera view.
        This cache can be reused for every thermal timestep.
        """
        H, W = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cache = ThermalViewCache()

        # CHANGED: explicit guard to keep type/None behavior sane
        if self.thermal_points_world is None:
            return cache

        # ---------------------------------------------------
        # Cache sparse thermal point projection / visibility
        # ---------------------------------------------------
        points_cam = self._world_to_camera(self.thermal_points_world, cam2world)
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]

        valid = Z < -1e-8
        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]
        valid_point_indices = np.nonzero(valid)[0]

        if len(X) > 0:
            depth_pts = -Z
            u = fx * X / depth_pts + cx
            v = fy * (-Y) / depth_pts + cy

            uu = np.round(u).astype(np.int32)
            vv = np.round(v).astype(np.int32)

            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu = uu[inside]
            vv = vv[inside]
            depth_pts = depth_pts[inside]
            valid_point_indices = valid_point_indices[inside]

            mesh_depth_at_points = depth_map[vv, uu]
            visible = np.isfinite(mesh_depth_at_points) & (
                depth_pts <= mesh_depth_at_points + self.thermal_depth_epsilon
            )

            cache.point_px = uu[visible]
            cache.point_py = vv[visible]
            cache.point_depth = depth_pts[visible]
            cache.point_linear_idx = valid_point_indices[visible]

        # ---------------------------------------------------
        # Cache dense mesh fill pixels and barycentric weights
        # ---------------------------------------------------
        if self.mesh_vertices_world is None or self.mesh_faces is None:
            return cache

        mesh_cam = self._world_to_camera(self.mesh_vertices_world, cam2world)
        VX = mesh_cam[:, 0]
        VY = mesh_cam[:, 1]
        VZ = mesh_cam[:, 2]

        if not np.any(VZ < -1e-8):
            return cache

        mesh_depth = -VZ
        proj_u = fx * VX / mesh_depth + cx
        proj_v = fy * (-VY) / mesh_depth + cy
        verts_2d = np.stack([proj_u, proj_v], axis=1)

        fill_px = []
        fill_py = []
        fill_vids = []
        fill_bary = []

        for f in self.mesh_faces:
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])

            z0, z1, z2 = VZ[i0], VZ[i1], VZ[i2]
            if z0 >= -1e-8 or z1 >= -1e-8 or z2 >= -1e-8:
                continue

            p0 = verts_2d[i0]
            p1 = verts_2d[i1]
            p2 = verts_2d[i2]

            d0 = -z0
            d1 = -z1
            d2 = -z2

            min_x = max(0, int(floor(min(p0[0], p1[0], p2[0]))))
            max_x = min(W - 1, int(ceil(max(p0[0], p1[0], p2[0]))))
            min_y = max(0, int(floor(min(p0[1], p1[1], p2[1]))))
            max_y = min(H - 1, int(ceil(max(p0[1], p1[1], p2[1]))))

            if min_x > max_x or min_y > max_y:
                continue

            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    if not np.isfinite(depth_map[py, px]):
                        continue

                    w0, w1, w2 = self._barycentric_coords_2d(
                        px + 0.5,
                        py + 0.5,
                        p0,
                        p1,
                        p2,
                    )
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue

                    tri_depth = w0 * d0 + w1 * d1 + w2 * d2
                    if abs(tri_depth - depth_map[py, px]) > self.thermal_depth_epsilon:
                        continue

                    fill_px.append(px)
                    fill_py.append(py)
                    fill_vids.append([i0, i1, i2])
                    fill_bary.append([w0, w1, w2])

        if fill_px:
            cache.mesh_px = np.asarray(fill_px, dtype=np.int32)
            cache.mesh_py = np.asarray(fill_py, dtype=np.int32)
            cache.mesh_vertex_ids = np.asarray(fill_vids, dtype=np.int32)
            cache.mesh_bary = np.asarray(fill_bary, dtype=np.float64)

        return cache

    def _render_thermal_image(
        self,
        point_temperatures: np.ndarray,
        mesh_vertex_temperatures: np.ndarray,
        depth_map: np.ndarray,
        cache: ThermalViewCache,
    ) -> np.ndarray:
        """
        Render thermal image using a precomputed fixed-view cache.
        Only temperature-dependent work happens here.
        """
        H, W = depth_map.shape

        # CHANGED: use actual configured background, not implicit black-only logic
        background = np.asarray(self.thermal_background, dtype=np.uint8)
        image = np.zeros((H, W, 3), dtype=np.uint8)
        image[:] = background

        # ---------------------------------------------------
        # Stage 1: sparse point splatting
        # ---------------------------------------------------
        if (
            cache.point_px is not None
            and cache.point_py is not None
            and cache.point_depth is not None
            and cache.point_linear_idx is not None
            and len(cache.point_px) > 0
        ):
            point_colors = self._temperature_to_rgb(point_temperatures[cache.point_linear_idx])
            z_buffer = np.full((H, W), np.inf, dtype=np.float64)
            radius = max(0, self.thermal_point_size // 2)

            if radius == 0:
                for px, py, pz, color in zip(cache.point_px, cache.point_py, cache.point_depth, point_colors):
                    md = depth_map[py, px]
                    if not np.isfinite(md):
                        continue
                    if pz > md + self.thermal_depth_epsilon:
                        continue
                    if pz < z_buffer[py, px]:
                        z_buffer[py, px] = pz
                        image[py, px] = color
            else:
                for px, py, pz, color in zip(cache.point_px, cache.point_py, cache.point_depth, point_colors):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            xx = px + dx
                            yy = py + dy
                            if 0 <= xx < W and 0 <= yy < H:
                                md = depth_map[yy, xx]
                                if not np.isfinite(md):
                                    continue
                                if pz > md + self.thermal_depth_epsilon:
                                    continue
                                if pz < z_buffer[yy, xx]:
                                    z_buffer[yy, xx] = pz
                                    image[yy, xx] = color

        # ---------------------------------------------------
        # Stage 2: dense mesh fill
        # ---------------------------------------------------
        if (
            cache.mesh_px is not None
            and cache.mesh_py is not None
            and cache.mesh_vertex_ids is not None
            and cache.mesh_bary is not None
            and len(cache.mesh_px) > 0
        ):
            tri_temps = np.sum(
                mesh_vertex_temperatures[cache.mesh_vertex_ids] * cache.mesh_bary,
                axis=1,
            )
            tri_colors = self._temperature_to_rgb(tri_temps)

            # CHANGED: fill only background-colored pixels, not only hard-coded black pixels
            empty_mask = np.all(image[cache.mesh_py, cache.mesh_px] == background, axis=1)
            image[cache.mesh_py[empty_mask], cache.mesh_px[empty_mask]] = tri_colors[empty_mask]

        return image

    def save_images(self):
        for subdir in ["images", "normals", "depths", "semantics", "instances", "thermal"]:
            (self.output_path / subdir).mkdir(exist_ok=True)

        for i in range(self.camera_idx):
            with h5py.File(self.output_path / f"{i}.hdf5", "r") as f:
                # rgb = np.array(f["colors"][:])
                # if self.enable_transparency:
                #     alpha = np.full((*rgb.shape[:2], 1), int(self.alpha * 255), dtype=np.uint8)
                #     rgb = np.concatenate([rgb[..., :3], alpha], axis=-1)
                # Image.fromarray(rgb).save(self.output_path / "images" / f"{i:04d}.png")

                normal = np.clip(f["normals"][:] * 255, 0, 255).astype(np.uint8)
                Image.fromarray(normal).save(self.output_path / "normals" / f"{i:04d}_normal.png")

                depth = np.array(f["depth"][:], dtype=np.float32)
                depth_mm = np.where(np.isfinite(depth), depth * 1000.0, 0.0).astype(np.uint16)
                Image.fromarray(depth_mm).save(self.output_path / "depths" / f"{i:04d}_depth.png")

                # semantic = f["category_id_segmaps"][:]
                # semantics = np.zeros((*semantic.shape, 3), dtype=np.uint8)
                # for j, color in self.color_map.items():
                #     semantics[semantic == int(j)] = color
                # Image.fromarray(semantics).save(self.output_path / "semantics" / f"{i:04d}_mask.png")

                # instance = f["instance_segmaps"][:]
                # instances = np.zeros((*instance.shape, 3), dtype=np.uint8)
                # for j, color in self.color_map.items():
                #     instances[instance == int(j)] = color
                # Image.fromarray(instances).save(self.output_path / "instances" / f"{i:04d}.png")

                if (
                    self.render_thermal
                    and self.thermal_points_world is not None
                    and self.thermal_values is not None
                ):
                    cam2world = np.asarray(self.metadata["frames"][i]["camera_to_world"], dtype=np.float64)
                    K = np.asarray(self.metadata["frames"][i]["intrinsics"], dtype=np.float64)

                    if i not in self.thermal_render_cache:
                        self.thermal_render_cache[i] = self._build_thermal_render_cache(
                            cam2world=cam2world,
                            K=K,
                            depth_map=depth,
                        )

                    cache = self.thermal_render_cache[i]

                    for j in range(0, self.thermal_values.shape[1], 2):
                        point_temperatures = self.thermal_values[:, j]

                        # CHANGED: use helper method for consistency and readability
                        mesh_vertex_temperatures = self._compute_mesh_vertex_temperatures(point_temperatures)

                        thermal = self._render_thermal_image(
                            point_temperatures=point_temperatures,
                            mesh_vertex_temperatures=mesh_vertex_temperatures,
                            depth_map=depth,
                            cache=cache,
                        )

                        Image.fromarray(thermal).save(
                            self.output_path / "thermal" / f"{i:04d}_thermal_{j:04d}.png"
                        )

            (self.output_path / f"{i}.hdf5").unlink()

    def save_metadata(self):
        with open(self.output_path / "meta_data.json", "w") as f:
            json.dump(self.metadata, f, indent=4)

    def run(self):
        poi = bproc.object.compute_poi(self.scene_objects)

        # CHANGED: guard against load_camera_path being None
        if self.load_camera_path is not None and self.load_camera_path.exists():
            with open(self.load_camera_path, "r") as f:
                camera_data = json.load(f)
            for frame in camera_data["frames"]:
                cam2world = np.asarray(frame["camera_to_world"], dtype=np.float64)
                self._add_camera_pose(cam2world)
        else:
            for _ in range(self.num_frames):
                self.generate_camera_pose(poi)

        bproc.renderer.set_output_format(enable_transparency=self.enable_transparency)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        render_data = bproc.renderer.render()
        bproc.writer.write_hdf5(str(self.output_path), render_data)
        self.save_metadata()
        self.save_images()


def main():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(BlenderProcRenderer).run()


if __name__ == "__main__":
    main()