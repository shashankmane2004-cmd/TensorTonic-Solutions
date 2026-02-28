import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T, dtype=float)
    pts = np.array(points, dtype=float)

    single_point = False
    if pts.ndim == 1:
        pts = pts[None, :]
        single_point = True

    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])              # (N,4)

    transformed_h = (T @ pts_h.T).T            # (N,4)
    transformed = transformed_h[:, :3]         # drop homogeneous coord

    if single_point:
        return transformed[0]
    return transformed