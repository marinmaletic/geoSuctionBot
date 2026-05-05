"""Sobel suction quality scoring.

Computes surface normals from the spatial gradient of the organised point
cloud image (treating XYZ channels as a 3-channel image and applying Sobel
derivative filters to obtain tangent vectors, then crossing them).
Flatness is measured the same way as the KNN method — local std of normals
in a pixel-space window.  Final score: flatness x approach_alignment.
"""

import numpy as np
from ur_suctionbot.knn import std_filt


def score_sobel(
    xyz: np.ndarray,        # (N, 3) point cloud
    u: np.ndarray,          # (N,) pixel x-coordinates of each point
    v: np.ndarray,          # (N,) pixel y-coordinates of each point
    H: int,                 # image height   
    W: int,                 # image width
    win: int,               # Std filter window size in pixels (should match cup footprint)
    approach: np.ndarray,   # (3,) unit vector — robot gripper approach direction
) -> np.ndarray:
    
    """Compute Sobel-based suction scores for every point.    """
    import cv2

    # Build an organised point cloud image: each pixel stores its 3-D position.
    pc_img = np.zeros((H, W, 3), dtype=np.float32)
    pc_img[v, u] = xyz.astype(np.float32)

    # Sobel filters on XYZ channels give tangent vectors along image axes.
    gy = cv2.Sobel(pc_img, cv2.CV_64F, 1, 0, ksize=5)  # tangent along columns
    gx = cv2.Sobel(pc_img, cv2.CV_64F, 0, 1, ksize=5)  # tangent along rows

    # Surface normal = cross product of the two tangent vectors.
    nrm      = np.cross(gx.reshape(-1, 3), gy.reshape(-1, 3))
    nrm_len  = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm      = np.where(nrm_len > 0, nrm / np.where(nrm_len > 0, nrm_len, 1.0), np.array([[0.0, 0.0, -1.0]]))
    nmap     = nrm.reshape(H, W, 3).astype(np.float32)

    mean_std = np.mean(std_filt(nmap, win), axis=2)

    mx       = mean_std.max()
    flatness = (1.0 - mean_std / mx) if mx > 1e-9 else np.ones((H, W), dtype=np.float32)

    flat_pt       = flatness[v, u]
    sobel_normals = nmap[v, u]
    approach_alignment = np.abs(sobel_normals @ approach)
    return (flat_pt * approach_alignment).astype(np.float32)