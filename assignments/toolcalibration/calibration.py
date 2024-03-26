import numpy as np
from scipy.linalg import svd


def pivot_calibration(transforms):
    """
    Pivot calibration

    Keyword arguments:
    transforms -- A list of 4x4 transformation matrices from the tracking system (Fi)
                  representing the tracked tool's position and orientation at
                  different instances.

    Returns:
    T          -- The calibration matrix T (in homogeneous coordinates) that defines
                  the offset (p_t) from the tracked part to the pivot point (tool tip).
    """

    ## TODO: Implement pivot calibration as discussed in the lecture

    # 1. extract rotation matrix R and translation vector p from each transformation matrix.
    # 2. construct matrix A and vector b
    # 3. solve for x in Ax = b
    
    T = np.eye(4) # identity matrix
    n = len(transforms) # number transfforms
    A = np.zeros((3*n, 6)) #  A
    b = np.zeros((3*n, 1)) # b
    for i, Fi in enumerate(transforms):
        Ri = Fi[:3, :3] # rotation matrix
        pi = Fi[:3, 3].reshape(3, 1) # translation vector 
        A[i*3:(i+1)*3, :3] = Ri 
        A[i*3:(i+1)*3, 3:] = -np.eye(3)
        b[i*3:(i+1)*3, :] = -pi
    U, s, V = svd(A, full_matrices=False)
    x = np.dot(V.T, np.dot(np.diag(1/s), np.dot(U.T, b)))
    T[:3, 3] = x[:3, 0] 
    return T


def calibration_device_calibration(camera_T_reference, camera_T_tool, reference_T_tip):
    """
    Tool calibration using calibration device

    Keyword arguments:
    camera_T_reference -- Transformation matrix from reference (calibration device) to camera.
    camera_T_tool      -- Transformation matrix from tool to camera.
    reference_T_tip    -- Transformation matrix from tip to reference (calibration device).

    Returns:
    T                  -- Calibration matrix from tool to tip.
    """

    ## TODO: Implement a calibration method which uses a calibration device
    T = np.eye(4)
    inv_camera_T_tool = np.linalg.inv(camera_T_tool) 
    # calibration T from tool to tip using given formula
    T = inv_camera_T_tool @ camera_T_reference @ reference_T_tip
    return T
