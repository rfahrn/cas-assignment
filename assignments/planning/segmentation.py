import numpy as np
from scipy import ndimage
import queue
from collections import deque


def region_grow(image, seed_point):
    """
    Performs a region growing on the image starting from 'seed_point'
    :param image: A 3D grayscale input image
    :param seed_point: The seed point for the algorithm
    :return: A 3D binary segmentation mask with the same dimensions as 'image'
    """
    segmentation_mask = np.zeros(image.shape, np.bool_)
    z, y, x = seed_point
    intensity = image[z, y, x]
    print(f'Image data at position ({x}, {y}, {z}) has intensity value {intensity}') # print intensity value of the seed point
    print('Computing region growing...', end='', flush=True)

    # test around with parameters to get a good result (tedious work, but it's the only way to get a good result)
    delta = 100  # higher - intesity range is wider (merging regions), lower - intensity range is narrower
    max_distance = 50 # increase to allow for larger regions to be merged
    
    ## choose a lower and upper threshold
    threshold_lower = max(intensity - delta, image.min())
    threshold_upper = min(intensity + delta, image.max())
    _segmentation_mask = (np.greater(image, threshold_lower)
                          & np.less(image, threshold_upper)).astype(np.bool_)

    ## pre-process the segmented image with a morphological filter
    structure = ndimage.generate_binary_structure(3, 1)
    _segmentation_mask = ndimage.binary_opening(_segmentation_mask, structure)

    to_check = deque()
    to_check.append((z, y, x))

    while to_check:
        z, y, x = to_check.popleft()

        if _segmentation_mask[z, y, x]:
            # Mark the current point as visited
            _segmentation_mask[z, y, x] = False
            segmentation_mask[z, y, x] = True

            # These for loops will visit all the neighbors of a voxel and see if
            # they belong to the region
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue    # Skip the center point
                        nz, ny, nx = z + dz, y + dy, x + dx

                        ## implement the code which checks whether the current
                        ## voxel (nz, ny, nx) belongs to the region or not
                        if (0 <= nz < image.shape[0] and 0 <= ny < image.shape[1] and 0 <= nx < image.shape[2] and _segmentation_mask[nz, ny, nx] and np.linalg.norm([nz - z, ny - y, nx - x]) <= max_distance): # check if  voxel is within  image and if it is not already visited
                                to_check.append((nz, ny, nx)) # add voxel to the queue

                        ## OPTIONALimplement a stop criteria such that the algorithm
                        ## doesn't check voxels which are too far away

    # Post-process the image with a morphological filter
    segmentation_mask = ndimage.binary_closing(segmentation_mask, structure=structure).astype(np.bool_)
    
    print('\rComputing region growing... [DONE]', flush=True)

    return segmentation_mask

