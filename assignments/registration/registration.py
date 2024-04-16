import numpy as np
from scipy.spatial import KDTree

def paired_point_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target point clouds.
    :param source: A N x 3 matrix with N 3D points.
    :param target: A N x 3 matrix with N 3D points.
    :return:
        T: 4x4 transformation matrix mapping source to target.
        R: 3x3 rotation matrix part of T.
        t: 1x3 translation vector part of T.
    """
    assert source.shape == target.shape
    T = np.eye(4)
    R = np.eye(3)
    t = np.zeros((1, 3))

    ## TODO: your code goes here
    
    # 1.) calc centroid of source and target point clouds
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    # 2.) center the source and target point clouds
    source_centered = source - centroid_source
    target_centered = target - centroid_target
    # 3.) calc covariance matrix
    M = source_centered.T @ target_centered
    # 4.) SVD on M
    U, W, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T # V @ U.T (so Vt.T = V)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # 6.) calc translation vector
    t = centroid_target - R @ centroid_source
    T[:3, :3] = R
    T[:3, 3] = t
    t = t.reshape(1, 3) # reshape to 1x3 vector as given
    return T, R, t


def get_initial_pose(source, target):
    """
    Calculates an initial rough registration or optionally returns a hand-picked initial pose.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: An initial 4 x 4 rigid transformation matrix.
    """
    T = np.eye(4)
    ## TODO: Your code goes here
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    translation = centroid_target - centroid_source
    T[:3, 3] = translation # set translation part of transformation matrix

    return T


def find_nearest_neighbor(source, target):
    """
    Finds the nearest neighbor in 'target' for every point in 'source'.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: A tuple containing two arrays: the first array contains the
             distances to the nearest neighbor in 'target' for each point
             in 'source', and the second array contains the indices of
             these nearest neighbors in 'target'.
    """
    ## TODO: replace this by your code use KDTree
    target_tree = KDTree(target)
    distances, indices = target_tree.query(source)
    return distances, indices

def icp(source, target, init_pose=None, max_iterations=10, tolerance=0.0001):
    """
    Iteratively finds the best transformation mapping the source points onto the target.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :param init_pose: Initial pose as a 4 x 4 transformation matrix.
    :param max_iterations: Maximum iterations.
    :param tolerance: Error tolerance.
    :return: The optimal 4 x 4 rigid transformation matrix, distances, and registration error.
    """

    # Initialisation
    T = np.eye(4) if init_pose is None else init_pose
    distances = 0
    error = np.finfo(float).max
    ## TODO: Your code goes here
    previous_error = error
    distances = np.zeros(source.shape[0])
    
    for _ in range(max_iterations):
        source_transformed = (T[:3, :3] @ source.T + T[:3, 3].reshape(3, 1)).T
        distances, indices = find_nearest_neighbor(source_transformed, target)
        matched_target = target[indices]
        T_current, _, _ = paired_point_matching(source_transformed, matched_target)
        T = T_current @ T
        current_error = np.mean(distances)
        if abs(previous_error - current_error) < tolerance:
            break
        previous_error = current_error

    return T, distances, current_error

