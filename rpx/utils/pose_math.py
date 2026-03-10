import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_matrix(pos, quat):
    """
    Converts position and quaternion to a 4x4 transformation matrix.
    Args:
        pos (array-like): [x, y, z]
        quat (array-like): [x, y, z, w]
    Returns:
        np.array: 4x4 transformation matrix
    """
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def compute_relative_transform(T1, T2):
    """
    Computes the relative transform from T1 to T2.
    T_rel = T1^-1 * T2
    """
    return np.linalg.inv(T1) @ T2

def matrix_to_pose(mat):
    """
    Converts a 4x4 transformation matrix back to position and quaternion.
    """
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat

def compute_rpe(pred_T_rel, gt_T_rel):
    """
    Computes Relative Pose Error (RPE) between predicted and ground truth relative transforms.
    Returns:
        trans_error: Euclidean distance
        rot_error: Geodesic distance (degrees)
    """
    # Translation error
    trans_error = np.linalg.norm(pred_T_rel[:3, 3] - gt_T_rel[:3, 3])
    
    # Rotation error
    rel_R = pred_T_rel[:3, :3].T @ gt_T_rel[:3, :3]
    rot_error = np.rad2deg(np.arccos(np.clip((np.trace(rel_R) - 1) / 2, -1, 1)))
    
    return trans_error, rot_error
