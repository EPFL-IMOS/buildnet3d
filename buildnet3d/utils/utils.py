import numpy as np
from typing import Union, List, Optional
from pathlib import Path
import pandas as pd

def build_translation(radius: float, theta: float, phi: float) -> np.ndarray:
    """Generate a 3D point on a sphere given radius, theta, and phi angles."""
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    
    return np.array([x, y, z])

def build_rotation_from_forward_vec(forward_vec: Union[np.ndarray, List[float]], 
                                    up_axis: np.ndarray = np.array([0, 0, 1]),
                                    inplane_rot: Optional[float] = None) -> np.ndarray:
    """
    Returns a camera rotation matrix for the given forward vector and up axis

    :param forward_vec: The forward vector which specifies the direction the camera.
    :param up_axis: The global up axis, typically [0, 1, 0].
    :param inplane_rot: The inplane rotation in radians. If None is given, the inplane rotation is determined only
                        based on the up vector.
    :return: A 3x3 camera rotation matrix.
    """
    # Normalize forward vector 
    f = forward_vec / np.linalg.norm(forward_vec)

    # Compute right vector [x-axis]
    r = np.cross(up_axis, f)
    r = r / np.linalg.norm(r)

    # Recompute up vector to ensure orthonormality
    u = np.cross(f, r)

    # Compose rotation matrix: columns are [right, up, -forward]
    R = np.stack((r, u, -f), axis=1)

    # Apply in-plane (roll) rotation if specified
    if inplane_rot is not None and abs(inplane_rot) > 1e-6:
        c, s = np.cos(inplane_rot), np.sin(inplane_rot)
        R_roll = np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1]
        ])
        R = R @ R_roll

    return R

def build_transformation(translation: Union[np.ndarray, List[float]],
                         rotation: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    """ Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: A 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat

def load_temperatures_from_csv(
        csv_path: Path,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        t_prefix: str = "T_",
        node_id_col: str = "node_id",
    ) -> tuple[np.ndarray, np.ndarray, list[float], Optional[np.ndarray]]:
        """
        Load thermal point cloud from CSV.
        Returns:
            points:      (N, 3)
            temps:       (N, T)
            timestamps:  list[T]
            node_ids:    (N,) or None
        """
        csv_path = Path(csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Thermal CSV does not exist: {csv_path}")
        df = pd.read_csv(csv_path)

        if node_id_col in df.columns:
            df = df.sort_values(by=node_id_col, kind="mergesort").reset_index(drop=True)
        node_ids = df[node_id_col].to_numpy() if node_id_col in df.columns else None

        required_xyz = [x_col, y_col, z_col]
        missing_xyz = [c for c in required_xyz if c not in df.columns]
        if missing_xyz:
            raise ValueError(f"Missing required coordinate columns in CSV: {missing_xyz}")

        temp_cols = [col for col in df.columns if col.startswith(t_prefix)]
        if not temp_cols:
            raise ValueError(
                f"No temperature columns found with prefix '{t_prefix}'. "
                f"Available columns: {list(df.columns)}"
            )

        def _sort_key(col_name: str):
            suffix = col_name[len(t_prefix):]
            try:
                return (0, float(suffix))
            except ValueError:
                return (1, suffix)

        temp_cols = sorted(temp_cols, key=_sort_key)

        timestamps = []
        for col in temp_cols:
            suffix = col[len(t_prefix):]
            try:
                timestamps.append(float(suffix))
            except ValueError:
                timestamps.append(float(len(timestamps)))

        points = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float64)
        temps = df[temp_cols].to_numpy(dtype=np.float64)

        valid_temp_mask = np.isfinite(temps).all(axis=1)
        if not np.all(valid_temp_mask):
            removed_count = int((~valid_temp_mask).sum())
            points = points[valid_temp_mask]
            temps = temps[valid_temp_mask]
            if node_ids is not None:
                node_ids = node_ids[valid_temp_mask]
        if len(points) == 0:
            raise ValueError("No valid points remain after removing NaN/Inf temperatures")

        return points, temps, timestamps, node_ids