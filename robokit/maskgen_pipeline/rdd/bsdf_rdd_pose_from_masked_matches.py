import argparse
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from time import time
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper

def draw_matches(ref_points, dst_points, img0, img1, inliers=None):
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]
    matches_mask = inliers.ravel() if inliers is not None else None
    return cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                           matchesMask=matches_mask, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8) * 255)

def load_intrinsics(path, img_shape):
    try:
        with open(path, 'r') as f:
            values = list(map(float, f.read().strip().split()))
        K = np.array(values).reshape(3, 3)
        if not (100 < K[0,0] < 5000 and 100 < K[1,1] < 5000 and
                0 < K[0,2] < img_shape[1] and 0 < K[1,2] < img_shape[0]):
            print("Warning: Invalid intrinsics, using default K")
            raise ValueError
    except:
        print("Warning: Failed to load intrinsics, using default K")
        K = np.array([[600, 0, img_shape[1]/2],
                      [0, 600, img_shape[0]/2],
                      [0, 0, 1]], dtype=np.float32)
    return K

def get_3d_centroid_from_mask_and_depth(mask, depth, K):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        raise ValueError(f"Too few mask points ({len(xs)})")
    depths = depth[ys, xs].astype(np.float32)
    valid = (depths > 100) & (depths < 5000)
    if np.sum(valid) < 10:
        raise ValueError(f"Too few valid depth values ({np.sum(valid)})")
    depths = depths[valid]
    xs = xs[valid]
    ys = ys[valid]
    z = np.median(depths) / 1000.0
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cx_img = np.mean(xs)
    cy_img = np.mean(ys)
    x = (cx_img - cx) * z / fx
    y = (cy_img - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)

def back_project(kp, depth, K):
    u, v = kp[:, 0], kp[:, 1]
    z = depth[v.astype(int), u.astype(int)] / 1000.0
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * z / fx

def align_3d_points(pts1, pts2):
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)
    pts1_centered = pts1 - centroid1
    pts2_centered = pts2 - centroid2
    H = pts1_centered.T @ pts2_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid2 - R @ centroid1
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def draw_object_axis(img, K, R, t, axis_length=0.05, label_axes=True, prefix="", debug=False):
    t = np.array(t).flatten().astype(np.float32)
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    axis = np.array([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    pts_2d, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, t, K, None)
    origin_2d = pts_2d[0].ravel().astype(int)
    x_2d = pts_2d[1].ravel().astype(int)
    y_2d = pts_2d[2].ravel().astype(int)
    z_2d = pts_2d[3].ravel().astype(int)
    h, w = img.shape[:2]
    points = [origin_2d, x_2d, y_2d, z_2d]
    for pt in points:
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            print(f"Warning: Axis point {pt} is outside image bounds ({w}, {h})")
            if debug:
                print(f"Projected points: origin={origin_2d}, x={x_2d}, y={y_2d}, z={z_2d}")
                print(f"Input: R=\n{R}\nt={t}\nK=\n{K}")
    cv2.line(img, tuple(origin_2d), tuple(x_2d), (0, 0, 255), 2)
    cv2.line(img, tuple(origin_2d), tuple(y_2d), (0, 255, 0), 2)
    cv2.line(img, tuple(origin_2d), tuple(z_2d), (255, 0, 0), 2)
    if label_axes:
        cv2.putText(img, f'{prefix}X', tuple(x_2d), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'{prefix}Y', tuple(y_2d), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f'{prefix}Z', tuple(z_2d), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img

def draw_object_axis_2d_fallback(img, R, center, axis_length=50, label_axes=True, prefix=""):
    center = np.array(center, dtype=np.int32)
    R_2d = R[:2, :2]
    axis = np.array([
        [axis_length, 0],
        [0, axis_length],
        [0, 0]
    ], dtype=np.float32)
    axis_rotated = (R_2d @ axis.T).T
    x_end = (center + axis_rotated[0]).astype(int)
    y_end = (center + axis_rotated[1]).astype(int)
    h, w = img.shape[:2]
    for pt in [center, x_end, y_end]:
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            print(f"Warning: 2D Axis point {pt} is outside image bounds ({w}, {h})")
    cv2.line(img, tuple(center), tuple(x_end), (0, 0, 255), 2)
    cv2.line(img, tuple(center), tuple(y_end), (0, 255, 0), 2)
    if label_axes:
        cv2.putText(img, f'{prefix}X', tuple(x_end), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'{prefix}Y', tuple(y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

def bundle_adjustment(T_0, T_1, pts3d_0, kp1, K, depth1, max_iterations=50):
    T_0_torch = torch.tensor(T_0, dtype=torch.float32, requires_grad=True)
    T_1_torch = torch.tensor(T_1, dtype=torch.float32, requires_grad=True)
    pts3d_0_torch = torch.tensor(pts3d_0, dtype=torch.float32)
    kp1_torch = torch.tensor(kp1, dtype=torch.float32)
    K_torch = torch.tensor(K, dtype=torch.float32)
    optimizer = torch.optim.Adam([T_0_torch, T_1_torch], lr=0.01)
    for _ in range(max_iterations):
        optimizer.zero_grad()
        pts3d_1_est = (T_1_torch @ torch.inverse(T_0_torch)) @ torch.cat([pts3d_0_torch, torch.ones(len(pts3d_0), 1)], dim=1).T
        uv1_est = (K_torch @ pts3d_1_est[:3, :]) / pts3d_1_est[2, :]
        error = torch.norm(uv1_est[:2, :].T - kp1_torch, dim=1)
        loss = torch.mean(error)  # Cauchy loss can be added
        loss.backward()
        optimizer.step()
    return T_0_torch.detach().numpy(), T_1_torch.detach().numpy()

def main(args):
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    mask_dir = scene_dir / "masks"
    depth_dir = scene_dir / "depth"
    K_path = scene_dir / "cam_K.txt"

    img0 = cv2.imread(str(rgb_dir / "000000.png"))
    img1 = cv2.imread(str(rgb_dir / "000001.png"))
    mask0 = cv2.imread(str(mask_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    mask1 = cv2.imread(str(mask_dir / "000001.png"), cv2.IMREAD_UNCHANGED)
    depth0 = cv2.imread(str(depth_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth1 = cv2.imread(str(depth_dir / "000001.png"), cv2.IMREAD_UNCHANGED)

    assert img0 is not None and img1 is not None, "Images not loaded"
    assert mask0 is not None and mask1 is not None, "Masks not loaded"
    assert depth0 is not None and depth1 is not None, "Depth images not loaded"

    mask0_bin = (mask0 > 0).astype(np.uint8)
    mask1_bin = (mask1 > 0).astype(np.uint8)
    img0_masked = apply_mask(img0, mask0_bin)
    img1_masked = apply_mask(img1, mask1_bin)
    K = load_intrinsics(K_path, img0.shape)
    print("Camera Intrinsics K:\n", K)

    RDD_model = build(weights='./weights/RDD-v2.pth')
    RDD_model.eval()
    RDD_wrap = RDD_helper(RDD_model)

    start = time()
    mkpts_0, mkpts_1, conf = RDD_wrap.match_lg(img0_masked, img1_masked)
    print(f"Found {len(mkpts_0)} matches in {time() - start:.2f} seconds")

    conf_threshold = 0.8
    valid = conf > conf_threshold
    mkpts_0 = mkpts_0[valid]
    mkpts_1 = mkpts_1[valid]
    print(f"Filtered to {len(mkpts_0)} high-confidence matches")

    pts3d_0, valid_0 = back_project(mkpts_0, depth0, K)
    pts3d_1, valid_1 = back_project(mkpts_1, depth1, K)
    valid = valid_0 & valid_1
    pts3d_0 = pts3d_0[valid]
    pts3d_1 = pts3d_1[valid]
    mkpts_0 = mkpts_0[valid]
    mkpts_1 = mkpts_1[valid]
    print(f"Valid 3D correspondences: {len(pts3d_0)}")

    T_01 = align_3d_points(pts3d_0, pts3d_1)
    T_0 = np.eye(4)
    T_1 = T_01  # Initial pose for frame 1

    # T_0, T_1 = bundle_adjustment(T_0, T_1, pts3d_0, mkpts_1, K, depth1)
    # R_0, t_0 = T_0[:3, :3], T_0[:3, 3]
    # R_1, t_1 = T_1[:3, :3], T_1[:3, 3]

    print("Frame 0 Pose:\n", T_0)
    print("Frame 1 Pose:\n", T_1)

    RT = np.eye(4)
    RT[:3, :3] = R_1
    RT[:3, 3] = t_1
    np.savetxt(scene_dir / "relative_pose_rt.txt", RT, fmt="%.6f")

    canvas = draw_matches(mkpts_0, mkpts_1, img0_masked, img1_masked, valid.astype(np.uint8))

    t_obj_0 = get_3d_centroid_from_mask_and_depth(mask0_bin, depth0, K)
    t_obj_1 = get_3d_centroid_from_mask_and_depth(mask1_bin, depth1, K)
    print(f"Frame 0 3D centroid: {t_obj_0}")
    print(f"Frame 1 3D centroid: {t_obj_1}")

    try:
        img0_axis = draw_object_axis(img0.copy(), K, R_0, t_obj_0, axis_length=0.05, label_axes=True, prefix="Ref", debug=True)
        img1_axis = draw_object_axis(img1.copy(), K, R_1, t_obj_1, axis_length=0.05, label_axes=True, prefix="Rel", debug=True)
    except Exception as e:
        print(f"Warning: 3D projection failed ({str(e)}), using 2D fallback")
        cx_0, cy_0 = K @ t_obj_0
        cx_1, cy_1 = K @ t_obj_1
        center_0 = np.array([cx_0[0]/cx_0[2], cy_0[1]/cy_0[2]], dtype=np.int32)
        center_1 = np.array([cx_1[0]/cx_1[2], cy_1[1]/cy_1[2]], dtype=np.int32)
        img0_axis = draw_object_axis_2d_fallback(img0.copy(), R_0, center_0, axis_length=50, label_axes=True, prefix="Ref")
        img1_axis = draw_object_axis_2d_fallback(img1.copy(), R_1, center_1, axis_length=50, label_axes=True, prefix="Rel")

    proj = K @ t_obj_0
    cx_0 = proj[0] / proj[2]
    cy_0 = proj[1] / proj[2]
    center_0 = np.array([cx_0[0]/cx_0[2], cy_0[1]/cy_0[2]], dtype=np.int32)
    center_1 = np.array([cx_1[0]/cx_1[2], cy_1[1]/cy_1[2]], dtype=np.int32)
    cv2.circle(img0_axis, tuple(center_0), 5, (255, 255, 0), -1)
    cv2.circle(img1_axis, tuple(center_1), 5, (255, 255, 0), -1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img0_axis[..., ::-1])
    axs[0].set_title("Frame 0: Reference Pose")
    axs[1].imshow(img1_axis[..., ::-1])
    axs[1].set_title("Frame 1: Relative Pose")
    axs[2].imshow(canvas[..., ::-1])
    axs[2].set_title("Object Mask Matches (Green: Inliers, Red: Outliers)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="Path to scene dir containing rgb/, masks/, depth/, cam_K.txt")
    parser.add_argument("--mode", default="match_lg", choices=["match", "match_dense", "match_lg", "match_3rd_party"])
    args = parser.parse_args()
    main(args)