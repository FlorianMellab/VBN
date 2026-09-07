import numpy as np
import cv2 as cv
from pathlib import Path

# Known physical radii (metres)
R_OUTER = 0.080   # 80mm
R_INNER = 0.066   # 66mm

# Known physical radii (metres) - EUTELSAT 16A (DOR CAD)
#Shall be confirmed
# R_OUTER = 0.597   # 597mm
# R_INNER = 0.553   # 550mm (Only an estimation....)

def inliers_to_circle_correspondence(
    inliers, xc, yc, a, b, theta, radius
):
    """
    Convert ellipse inlier points into 2D–3D correspondences.

    Input:
        inliers : (N,2) image points (global coords)
        xc, yc  : ellipse center (global)
        a, b    : ellipse semi-axes
        theta   : ellipse rotation [rad]
        radius  : 3D circle radius

    Output:
        object_points : (N,3)
        image_points  : (N,2)
    """

    pts = inliers.astype(np.float64).copy()

    # --- 1. Translate to center ---
    pts[:, 0] -= xc
    pts[:, 1] -= yc

    # --- 2. Rotate into ellipse canonical frame ---
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    x = cos_t * pts[:, 0] - sin_t * pts[:, 1]
    y = sin_t * pts[:, 0] + cos_t * pts[:, 1]

    # --- 3. Normalise by ellipse axes ---
    # This maps ellipse → unit circle
    x_norm = x / a
    y_norm = y / b

    # --- 4. Compute angle parameter ---
    angles = np.arctan2(y_norm, x_norm)

    # --- 5. Map to 3D circle ---
    object_points = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros_like(angles)
    ], axis=1)

    return object_points.astype(np.float32), inliers.astype(np.float32)

def solve_pose(object_points, image_points, K):
    solutions = cv.solvePnPGeneric(
        object_points,
        image_points,
        K,
        distCoeffs=None,
        flags=cv.SOLVEPNP_IPPE
    )

    _, rvecs, tvecs, reproj_errors = solutions

    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, reproj_errors):
        if tvec[2, 0] > 0:
            R, _ = cv.Rodrigues(rvec)
            best = (rvec, tvec, R, float(err.squeeze()))
            break

    if best is None:
        rvec = rvecs[0]
        tvec = tvecs[0]
        R, _ = cv.Rodrigues(rvec)
        reproj_errors = np.asarray(reproj_errors).astype(float).ravel
        best = (rvec, tvec, R, reproj_errors)

    return best

def rotation_to_rpy(R):
    """
    Convert rotation matrix to roll, pitch, yaw (degrees).
    Convention: roll = rotation about z, pitch = rotation about x,
    yaw = rotation about y (pitch <-> y_drift, yaw <-> x_drift).
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2]) 
        yaw   = np.arctan2(-R[2,0], sy)      
        roll  = np.arctan2(R[1,0], R[0,0])   
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = 0

    return np.degrees([roll, pitch, yaw])

def compute_pose(ellipse_data, K, R_OUTER, R_INNER):
    """
    Compute pose for a single frame from its ellipse_data dict (the same
    structure the ellipse-detection pipeline saves to *_ellipses.npy).

    This is the per-frame core of what the old __main__ loop did, pulled
    out so it can be called either standalone (batch, over saved .npy
    files) or inline, immediately after ellipse detection saves a frame's
    ellipse_data, so tracking mode has pose available for the very next
    frame in the same run.

    Input:
        ellipse_data: dict with "outer"/"inner" keys, each containing
                      xc, yc, a, b, theta_deg, inliers, x_offset, y_offset
                      (cropped-frame coordinates + offsets, as saved by
                      the ellipse detection script).
        K:            3x3 camera intrinsics matrix.
        R_OUTER:      outer ring radius in metres.
        R_INNER:      inner ring radius in metres.

    Output:
        pose_data: dict with "outer"/"inner" keys, each containing
                   tvec, R, reprojection_error — same format previously
                   saved to *_ellipses_pose.npy.
    """
    outer = ellipse_data["outer"]
    inner = ellipse_data["inner"]

    outer_xc = outer["xc"] + outer["x_offset"]
    outer_yc = outer["yc"] + outer["y_offset"]
    inner_xc = inner["xc"] + inner["x_offset"]
    inner_yc = inner["yc"] + inner["y_offset"]

    inliers_outer = outer["inliers"] + np.array([outer["x_offset"], outer["y_offset"]])
    inliers_inner = inner["inliers"] + np.array([inner["x_offset"], inner["y_offset"]])

    obj_outer, img_outer = inliers_to_circle_correspondence(
        inliers_outer, outer_xc, outer_yc, outer["a"], outer["b"],
        np.radians(outer["theta_deg"]), R_OUTER
    )
    obj_inner, img_inner = inliers_to_circle_correspondence(
        inliers_inner, inner_xc, inner_yc, inner["a"], inner["b"],
        np.radians(inner["theta_deg"]), R_INNER
    )

    rvec_o, tvec_o, R_o, _ = solve_pose(obj_outer, img_outer, K)
    proj_o, _ = cv.projectPoints(obj_outer, rvec_o, tvec_o, K, None)
    reproj_o = float(np.linalg.norm(proj_o.squeeze() - img_outer, axis=1).mean())

    rvec_i, tvec_i, R_i, _ = solve_pose(obj_inner, img_inner, K)
    proj_i, _ = cv.projectPoints(obj_inner, rvec_i, tvec_i, K, None)
    reproj_i = float(np.linalg.norm(proj_i.squeeze() - img_inner, axis=1).mean())

    # print("Outer ring pose: ", tvec_o)

    return {
        "outer": {"tvec": tvec_o, "R": R_o, "reprojection_error": reproj_o},
        "inner": {"tvec": tvec_i, "R": R_i, "reprojection_error": reproj_i},
    }

if __name__ == "__main__":
    ClearSpace = False

    if ClearSpace:
        # Camera parameters - ClearSpace. From the test report:
        # For the VIS camera model, we describe the 6 mm lens / 50.8° CFOV, 2590x1942 detector, 2.2 μm
        # pitch, and the visible-band exposure and noise model that the Camera Model pipeline applies
        # (photon + readout noise, gain, irradiance) stage before readout.
        image_size = [2590, 1942]
        square_fov_deg = 50.8

        fx = (image_size[0] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)
        fy = (image_size[1] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)

    else:
        # Camera parameters - AuricamD80
        image_size = [2048, 2048]
        # image_size = [1024, 1024]
        square_fov_deg = 56.6

        fx = (image_size[0] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)
        fy = (image_size[1] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)

        print(square_fov_deg)
        print(fx)


    cx = image_size[0] / 2
    cy = image_size[1] / 2 


    dataset_dir = Path(r"results\updated_model_06072026\CPO_dataset_2")
    output_dir  = Path(r"results\updated_model_06072026\CPO_dataset_2_pose")
    output_dir.mkdir(exist_ok=True)

    data_extensions = {".npy"}
    files = [f for f in dataset_dir.iterdir() if f.suffix.lower() in data_extensions]

    if not files:
        print(f"No ellipse files found in {dataset_dir}")

    for file in files:
        print(f"\n{'='*60}\nProcessing: {file.name}\n{'='*60}")
        stem = file.stem

        # Camera intrinsics
        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ], dtype=np.float64)

        # Load saved ellipse dict (saved with np.save as a pickle dict)
        ellipse_data = np.load(
            file,
            allow_pickle=True
        ).item()  # .item() converts 0-d object array back to a Python dict

        outer = ellipse_data["outer"]
        inner = ellipse_data["inner"]

        outer_xc = outer["xc"] + outer["x_offset"]
        outer_yc = outer["yc"] + outer["y_offset"]

        inner_xc = inner["xc"] + inner["x_offset"]
        inner_yc = inner["yc"] + inner["y_offset"]

        inliers_outer = outer["inliers"] + np.array([outer["x_offset"], outer["y_offset"]])
        inliers_inner = inner["inliers"] + np.array([inner["x_offset"], inner["y_offset"]])

        # Sample image points from both ellipses
        obj_outer, img_outer = inliers_to_circle_correspondence(
            inliers_outer,
            outer_xc, outer_yc,
            outer["a"], outer["b"],
            np.radians(outer["theta_deg"]),
            R_OUTER
        )

        obj_inner, img_inner = inliers_to_circle_correspondence(
            inliers_inner,
            inner_xc, inner_yc,
            inner["a"], inner["b"],
            np.radians(inner["theta_deg"]),
            R_INNER
        )

        # Stack both rings — more points, better constrained
        image_points  = np.vstack([img_outer, img_inner])
        object_points = np.vstack([obj_outer, obj_inner])


        #===============================
        # Solve pose OUTER ONLY
        # ===============================
        rvec_o, tvec_o, R_o, err_o = solve_pose(obj_outer, img_outer, K)

        proj_o, _ = cv.projectPoints(obj_outer, rvec_o, tvec_o, K, None)
        proj_o = proj_o.squeeze()
        reproj_o = np.linalg.norm(proj_o - img_outer, axis=1).mean()

        # ===============================
        # Solve pose INNER ONLY
        # ===============================
        rvec_i, tvec_i, R_i, err_i = solve_pose(obj_inner, img_inner, K)

        proj_i, _ = cv.projectPoints(obj_inner, rvec_i, tvec_i, K, None)
        proj_i = proj_i.squeeze()
        reproj_i = np.linalg.norm(proj_i - img_inner, axis=1).mean()
        
        poses = [
            ("outer", reproj_o, rvec_o, tvec_o, R_o),
            ("inner", reproj_i, rvec_i, tvec_i, R_i),
        ]

        best_label, best_err, rvec_best, tvec_best, R_best = min(poses, key=lambda x: x[1])
        
        print("\n===== BEST POSE =====")
        print(f"Selected: {best_label}")
        print(f"Reprojection error: {best_err:.2f} px")
        print(f"tvec: {tvec_best.ravel()}")
        print(f"R:\n{R_best}")

        pose_data = {
            "outer": {
                "tvec": tvec_o,
                "R": R_o,
                "reprojection_error": reproj_o
            },
            "inner": {
                "tvec": tvec_i,
                "R": R_i,
                "reprojection_error": reproj_i
            }
        }

        np.save(output_dir / f"{stem}_pose.npy", pose_data)