import numpy as np
import cv2 as cv
from pathlib import Path

# Known physical radii (metres)
R_OUTER = 0.080*7.4   # 80mm
R_INNER = 0.066*7.4   # 66mm

# Known physical radii (metres) - EUTELSAT 16A (DOR CAD)
#Shall be confirmed
# R_OUTER = 0.597   # 597mm
# R_INNER = 0.553   # 550mm (Only an estimation....)

def inliers_to_circle_correspondence_legacy(
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

def ellipse_model_to_circle_correspondence(
    xc,
    yc,
    a,
    b,
    theta,
    radius,
    n_points=360,
    endpoint=False,
):
    """
    Uniformly sample a complete fitted ellipse and associate each sampled
    image point with a point on a physical circle of known radius.

    This avoids biasing PnP toward whichever ellipse arc happened to be
    visible or survived RANSAC.

    Parameters
    ----------
    xc, yc : float
        Ellipse centre in full-image coordinates.

    a, b : float
        Ellipse semi-axis lengths in pixels, using the same parameterization
        as skimage.measure.EllipseModel.

    theta : float
        Ellipse orientation in radians.

    radius : float
        Physical circle radius in metres.

    n_points : int
        Number of uniformly distributed samples around the complete model.

    endpoint : bool
        Whether to repeat the zero-angle sample at 2*pi. This should normally
        remain False because PnP does not need the duplicated point.

    Returns
    -------
    object_points : ndarray, shape (N, 3), float32
        Uniform physical-circle points in the local XY plane.

    image_points : ndarray, shape (N, 2), float32
        Uniform samples of the fitted image ellipse.
    """
    if n_points < 4:
        raise ValueError(
            "At least four model samples are required."
        )

    if a <= 0 or b <= 0:
        raise ValueError(
            f"Ellipse axes must be positive, received a={a}, b={b}."
        )

    if radius <= 0:
        raise ValueError(
            f"Circle radius must be positive, received {radius}."
        )

    phi = np.linspace(
        0.0,
        2.0 * np.pi,
        n_points,
        endpoint=endpoint,
        dtype=np.float64,
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # skimage EllipseModel parameterization:
    #
    # x = xc + a*cos(theta)*cos(phi) - b*sin(theta)*sin(phi)
    # y = yc + a*sin(theta)*cos(phi) + b*cos(theta)*sin(phi)
    image_x = (
        xc
        + a * cos_theta * cos_phi
        - b * sin_theta * sin_phi
    )

    image_y = (
        yc
        + a * sin_theta * cos_phi
        + b * cos_theta * sin_phi
    )

    image_points = np.column_stack([
        image_x,
        image_y,
    ])

    object_points = np.column_stack([
        radius * cos_phi,
        radius * sin_phi,
        np.zeros_like(phi),
    ])

    return (
        object_points.astype(np.float32),
        image_points.astype(np.float32),
    )

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

def compute_pose_legacy(ellipse_data, K, R_OUTER, R_INNER):
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

    obj_outer, img_outer = inliers_to_circle_correspondence_legacy(
        inliers_outer, outer_xc, outer_yc, outer["a"], outer["b"],
        np.radians(outer["theta_deg"]), R_OUTER
    )
    obj_inner, img_inner = inliers_to_circle_correspondence_legacy(
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


def compute_pose(
    ellipse_data,
    K,
    R_OUTER,
    R_INNER,
    n_model_points=360
):
    """
    Recover outer and inner LAR poses from uniformly sampled complete ellipse
    models instead of from the potentially one-sided distribution of RANSAC
    inliers.

    This specifically removes the direct dependence of PnP on angular
    coverage of the observed arc.

    Parameters
    ----------
    ellipse_data : dict
        Dictionary containing "outer" and "inner" ellipse models.

    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.

    R_OUTER, R_INNER : float
        Physical outer and inner LAR radii in metres.

    n_model_points : int
        Number of uniform full-ellipse samples used for each PnP solve.

    Returns
    -------
    pose_data : dict
        Compatible outer/inner pose data
    """
    K = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(3, 3)

    required_labels = [
        "outer",
        "inner",
    ]

    for label in required_labels:
        if label not in ellipse_data:
            raise KeyError(
                f"ellipse_data is missing '{label}'. "
                f"Available keys: {list(ellipse_data.keys())}"
            )

    outer = ellipse_data["outer"]
    inner = ellipse_data["inner"]

    required_fields = [
        "xc",
        "yc",
        "a",
        "b",
        "theta_deg",
        "x_offset",
        "y_offset",
    ]

    for label, ellipse in [
        ("outer", outer),
        ("inner", inner),
    ]:
        missing = [
            field
            for field in required_fields
            if field not in ellipse
        ]

        if missing:
            raise KeyError(
                f"{label} ellipse is missing fields: {missing}"
            )

    # Full-image ellipse centres.
    outer_xc = (
        float(outer["xc"])
        + float(outer["x_offset"])
    )

    outer_yc = (
        float(outer["yc"])
        + float(outer["y_offset"])
    )

    inner_xc = (
        float(inner["xc"])
        + float(inner["x_offset"])
    )

    inner_yc = (
        float(inner["yc"])
        + float(inner["y_offset"])
    )

    # -------------------------------------------------------------------------
    # Uniform complete-model sampling
    # -------------------------------------------------------------------------
    obj_outer, img_outer = (
        ellipse_model_to_circle_correspondence(
            xc=outer_xc,
            yc=outer_yc,
            a=float(outer["a"]),
            b=float(outer["b"]),
            theta=np.radians(
                float(outer["theta_deg"])
            ),
            radius=float(R_OUTER),
            n_points=n_model_points,
            endpoint=False,
        )
    )

    obj_inner, img_inner = (
        ellipse_model_to_circle_correspondence(
            xc=inner_xc,
            yc=inner_yc,
            a=float(inner["a"]),
            b=float(inner["b"]),
            theta=np.radians(
                float(inner["theta_deg"])
            ),
            radius=float(R_INNER),
            n_points=n_model_points,
            endpoint=False,
        )
    )

    # -------------------------------------------------------------------------
    # Outer model pose
    # -------------------------------------------------------------------------
    (
        rvec_outer,
        tvec_outer,
        R_outer,
        reprojection_outer,
    ) = solve_pose(
        obj_outer,
        img_outer,
        K,
    )

    # -------------------------------------------------------------------------
    # Inner model pose
    # -------------------------------------------------------------------------
    (
        rvec_inner,
        tvec_inner,
        R_inner,
        reprojection_inner,
    ) = solve_pose(
        obj_inner,
        img_inner,
        K,
    )

    return {
        "outer": {
            "rvec": rvec_outer,
            "tvec": tvec_outer,
            "R": R_outer,
            "reprojection_error": float(
                reprojection_outer
            ),
            "correspondence_source": (
                "uniform_complete_ellipse_model"
            ),
            "n_model_points": int(
                n_model_points
            ),
        },
        "inner": {
            "rvec": rvec_inner,
            "tvec": tvec_inner,
            "R": R_inner,
            "reprojection_error": float(
                reprojection_inner
            ),
            "correspondence_source": (
                "uniform_complete_ellipse_model"
            ),
            "n_model_points": int(
                n_model_points
            ),
        },
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


    dataset_dir = Path(r"results\final_model_28082026\CPO_dataset_sensitivity_panels_rotated_scaled")
    output_dir  = Path(r"results\final_model_28082026\CPO_dataset_sensitivity_panels_rotated_scaled_pose")
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

        pose_data = compute_pose(
            ellipse_data,
            K,
            R_OUTER,
            R_INNER,
            n_model_points=360,
        )

        np.save(output_dir / f"{stem}_pose.npy", pose_data)
