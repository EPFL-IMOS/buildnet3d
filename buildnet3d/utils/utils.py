import numpy as np
from typing import Union, List, Optional

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