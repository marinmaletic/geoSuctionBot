"""RANSAC suction quality scoring.

For each point, all neighbours within a physically-sized sphere (radius =
cup radius in metres) are collected. Random triplets of neighbours are
sampled and plane hypotheses are tested; the hypothesis with the most
inliers wins.

Final score: flatness x approach_alignment
"""

import sys
sys.modules.setdefault("open3d.ml", type(sys)("open3d.ml"))
import numpy as np
import open3d as o3d


def gpu_available() -> bool:
    return False


def score_ransac(
    xyz: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    H: int,
    W: int,
    win: int,
    cup_size_mm: float | None,
    iters: int,
    tol_mm: float,
    approach: np.ndarray,
    fx: float,
    use_gpu: bool = False,
) -> np.ndarray:
    import cv2, time
    t0 = time.perf_counter()

    if cup_size_mm is not None:
        radius = (cup_size_mm / 2.0) / 1000.0
    else:
        med_z  = float(np.median(xyz[:, 2]))
        radius = (win / 2.0) * med_z / fx

    dist_thr = tol_mm / 1000.0
    min_nb   = 20
    n        = len(xyz)
    rng      = np.random.default_rng(42)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    tree = o3d.geometry.KDTreeFlann(pcd)

    nb_lists = []
    for i in range(n):
        _, idx, _ = tree.search_radius_vector_3d(xyz[i], radius)
        nb_lists.append(list(idx))

    inlier_ratio = np.zeros(n, dtype=np.float32)
    best_normals = np.zeros((n, 3), dtype=np.float32)
    best_normals[:, 2] = 1.0

    for i in range(n):
        idx = nb_lists[i]
        m   = len(idx)
        if m < min_nb:
            continue

        nb   = xyz[idx]
        nb_c = nb - nb.mean(axis=0)

        tri   = rng.integers(0, m, size=(iters, 3))
        s     = nb[tri]
        v1    = s[:, 1] - s[:, 0]
        v2    = s[:, 2] - s[:, 0]
        nrms  = np.cross(v1, v2)
        lens  = np.linalg.norm(nrms, axis=1)
        valid = lens > 1e-9
        if not valid.any():
            continue
        nrms[valid]  /= lens[valid, None]
        nrms[~valid]  = 0.0

        dists  = np.abs(nb_c @ nrms[valid].T)
        counts = (dists < dist_thr).sum(axis=0)
        best_j   = int(np.argmax(counts))
        best_cnt = int(counts[best_j])

        inlier_ratio[i] = best_cnt / m

        best_n = nrms[valid][best_j]
        if np.dot(best_n, approach) < 0:
            best_n = -best_n
        best_normals[i] = best_n

    approach_alignment = np.abs(best_normals @ approach).astype(np.float32)
    scores = (inlier_ratio * approach_alignment).astype(np.float32)

    # Soft edge penalty — weight linearly from 0 at the object boundary
    # up to 1.0 at one cup radius distance from the edge.
    valid_mask       = np.zeros((H, W), dtype=np.uint8)
    valid_mask[v, u] = 1
    dist_map         = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
    med_z            = float(np.median(xyz[:, 2]))
    radius_px        = (radius * fx) / med_z
    weight           = (dist_map[v, u] / radius_px).clip(0.0, 1.0).astype(np.float32)
    scores           = scores * weight

    return scores