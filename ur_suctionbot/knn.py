"""KNN suction quality scoring.

Estimates a surface normal at each point via PCA of its k nearest 3-D
neighbours, then measures how consistently those normals point in the same
direction across a pixel-space window (flatness).  Final score:
    score = flatness x approach_alignment
where `approach_alignment = |normal · approach|` and `approach` is the robot's
gripper approach direction.
"""

import sys
sys.modules.setdefault("open3d.ml", type(sys)("open3d.ml"))
import numpy as np
import open3d as o3d


def std_filt(img: np.ndarray, wlen: int) -> np.ndarray:
    """Per-pixel local standard deviation over a square window.

    Uses the identity std = sqrt(E[x²] - E[x]²) computed with two box filter passes.
    """
    import cv2
    ksize = (wlen | 1, wlen | 1)
    img   = img.astype(np.float32)
    mean  = cv2.boxFilter(img,       -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.boxFilter(img * img, -1, ksize, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.abs(mean2 - mean * mean))


def estimate_normals_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """Estimate per-point surface normals using PCA of k nearest neighbours.

    The covariance matrix of each local neighbourhood is decomposed; the
    eigenvector for the smallest eigenvalue is the surface normal.  All
    normals are oriented to face the camera.
    """
    toward_camera = np.array([0.0, 0.0, -1.0])
    n = len(xyz)
    k = min(k, n - 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    tree = o3d.geometry.KDTreeFlann(pcd)

    idx_arr = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        _, idx, _ = tree.search_knn_vector_3d(xyz[i], k + 1)
        idx_arr[i] = idx[1:]

    nb      = xyz[idx_arr] - xyz[:, None, :]          # (N, k, 3) offsets
    cov     = np.einsum("nki,nkj->nij", nb, nb)        # (N, 3, 3) covariance
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]                            # smallest eigenvector

    flip = (normals @ toward_camera) < 0
    normals[flip] = -normals[flip]
    return normals.astype(np.float32)


def score_knn(
    xyz: np.ndarray,    # (N, 3) point cloud
    u: np.ndarray,      # (N,) pixel x-coordinates of each point
    v: np.ndarray,    # (N,) pixel y-coordinates of each point
    H: int,             # image height
    W: int,        # image width
    k: int,             # KNN neighbour count for normal estimation
    win: int,       # Std filter window size in pixels (should match cup footprint)
    approach: np.ndarray,   # (3,) unit vector — robot gripper approach direction
) -> np.ndarray:
    """Compute KNN-based suction scores for every point.    """
    normals = estimate_normals_knn(xyz, k)

    # Scatter normals into image space to apply the 2-D std filter.
    nmap = np.zeros((H, W, 3), dtype=np.float32)
    nmap[v, u] = normals

    mean_std = np.mean(std_filt(nmap, win), axis=2)

    mx       = mean_std.max()
    flatness = (1.0 - mean_std / mx) if mx > 1e-9 else np.ones((H, W), dtype=np.float32)

    flat_pt = flatness[v, u]
    approach_alignment  = np.abs(normals @ approach)
    return (flat_pt * approach_alignment).astype(np.float32)