
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from skimage.measure import EllipseModel, ransac
import warnings
from itertools import combinations
import re
import matplotlib.cm as cm
from skimage.feature import canny

from pose_estimation import compute_pose

DATASET_CONFIGS = {
    "CPO": dict(
        dataset_dir=Path(r"CPO_dataset_sensitivity_panels_rotated_fullscale_2_restricted"),
        output_dir=Path(r"results/final_model_28082026/CPO_dataset_sensitivity_panels_rotated_fullscale_2_restricted"),
        R_OUTER=0.080, R_INNER=0.066, 
        dmin=1.5, dmax=4.0, 
        min_curvature=0.3, 
        tracking_ellipse_dir=None, 
        tracking_pose_dir=None, 
        frame_prefix=None, 
        camera="AuricamD80", 
    ),
    "DOR": dict(
        dataset_dir=Path(r"DOR_dataset/rendering"),
        output_dir=Path(r"results/final_model_28082026/DOR_dataset"),
        R_OUTER=0.597, R_INNER=0.553,
        dmin=1.5, dmax=4.0,
        min_curvature=0.2,
        tracking_ellipse_dir=Path(r"results/final_model_28082026/DOR_dataset"),
        tracking_pose_dir=Path(r"results/final_model_28082026/DOR_dataset_pose"),
        frame_prefix="WCAM1_AURICAM",
        camera="AuricamD80",
        #DOR dataset specific tuning
        canny_low = 60, 
        curvature_window=11, 
    ),
    "CPO_scaled": dict(
        dataset_dir=Path(r"CPO_dataset_sensitivity_panels_rotated_scaled"),
        output_dir=Path(r"results/final_model_28082026/CPO_dataset_sensitivity_panels_rotated_scaled"),
        R_OUTER=0.080*7.4, R_INNER=0.066*7.4,
        dmin=1.5, dmax=4.0,
        min_curvature=0.2,
        tracking_ellipse_dir=None,
        tracking_pose_dir=None,
        frame_prefix=None,
        camera="AuricamD80",

    ),
    "CPO_dataset_X": dict(
        dataset_dir=Path(r"CPO_dataset_X\1 - DATA\Session_2\shadows\batch_SDW1\run2\raws_tracking"),
        output_dir=Path(r"results/final_model_28082026/CPO_dataset_X"),
        R_OUTER=0.080, R_INNER=0.066,
        dmin=1.0, dmax=4.0,
        min_curvature=0.3,
        tracking_ellipse_dir=None,
        tracking_pose_dir=None,
        frame_prefix=None,
        camera="camera_X",
    ),
    "CPO_dataset_X_tracking": dict(
        dataset_dir=Path(r"CPO_dataset_X\1 - DATA\Session_2\shadows\batch_SDW1\run2\raws_tracking"),
        output_dir=Path(r"results/final_model_28082026/CPO_dataset_X_tracking"),
        R_OUTER=0.080, R_INNER=0.066,
        dmin=1.0, dmax=4.0,
        min_curvature=0.3,
        tracking_ellipse_dir=Path(r"results/final_model_28082026/CPO_dataset_X_tracking"),
        tracking_pose_dir=Path(r"results/final_model_28082026/CPO_dataset_X_tracking_pose"),
        frame_prefix="dataset_X_scene",
        camera="camera_X",
    ),
}

DATASET = "CPO_dataset_X_tracking"  # Change this to switch datasets
cfg = DATASET_CONFIGS[DATASET]
sma_ratio_nominal = cfg["R_INNER"] / cfg["R_OUTER"]

# Known physical radii (metres) - GEO sat model
# R_OUTER = 0.080   # 80mm
# R_INNER = 0.066   # 66mm

# Known physical radii (metres) - EUTELSAT (DOR CAD)
# R_OUTER = 0.597   # 597mm
# R_INNER = 0.553   # 553mm

#####
#Helper functions
#####
def read_image(file):
    file = Path(file)
    if file.suffix.lower() in (".tiff", ".tif"):
        img = tifffile.imread(file)
        green = img[:, :, 1]  # uint16, 12-bit range 0-4095
        return green
    else:
        img = cv.imread(str(file), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {file}")
        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img.astype(np.uint16)  # keep dtype consistent with TIFF path


######
#Contour map
######
def find_edges(img, low, high):
    if img.max() > 255:
        img_8bit = (img / 16).astype(np.uint8)  # 12-bit → 8-bit
    else:
        img_8bit = img.astype(np.uint8)          # already 8-bit
    edges = cv.Canny(img_8bit, low, high)
    return edges

#Testing with Skimage's 12-bit edge detector
def find_edges_skimage(img, sigma=1.0, low_threshold=None, high_threshold=None):
    img_float = img.astype(np.float64)
    edges = canny(img_float, sigma=sigma,
                  low_threshold=low_threshold, high_threshold=high_threshold)
    return (edges * 255).astype(np.uint8)  # binary edge map, same format as your find_edges()

def find_contours(seg):
    contours, _ = cv.findContours(
        seg,                     
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE
    )
    return contours

#####
#Filtering
#####
def refine_search_area_symmetric(edges, kernel_size, density_threshold, expected_max_radius_px, margin):
    """
    Like refine_search_area, but forces a symmetric crop sized by known
    expected object size, rather than letting the crop shape itself around
    wherever edges happen to be strongest (which biases against the
    shadowed side under asymmetric illumination).
    """
    h, w = edges.shape
    edge_float = (edges > 0).astype(np.float32)
    density_map = cv.boxFilter(edge_float, ddepth=-1, ksize=(kernel_size, kernel_size), normalize=True)
    dense_mask = (density_map > density_threshold).astype(np.uint8)

    if not np.any(dense_mask):
        print("Warning: no dense region found, returning full edge map.")
        return edges, (0, 0, w, h)

    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dense_mask)
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])

    # Use the CENTROID of the densest region as a seed point, not its bounding box
    cx, cy = centroids[largest_label]

    half_size = expected_max_radius_px + margin
    x1 = max(0, int(cx - half_size))
    y1 = max(0, int(cy - half_size))
    x2 = min(w, int(cx + half_size))
    y2 = min(h, int(cy + half_size))

    cropped_edges = edges[y1:y2, x1:x2]
    return cropped_edges, (x1, y1, x2, y2)


def curvature_signs(contours, window):
    """
    Returns a list of sign arrays, one per contour.
    Each array has +1 (left turn) or -1 (right turn) at every point.
    Majority-vote smoothing over a sliding window suppresses noise-induced
    spurious sign changes along otherwise smooth arcs.
    """
    all_signs = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        n = len(pts)
        signs = np.zeros(n, dtype=np.int8)
        for i in range(1, n - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            signs[i] = 1 if cross >= 0 else -1
        if n >= 2:
            signs[0] = signs[1]
            signs[-1] = signs[-2]

        # Majority vote over sliding window
        smoothed = signs.copy()
        half = window // 2
        for i in range(half, n - half):
            neighbourhood = signs[i - half : i + half + 1]
            smoothed[i] = 1 if np.sum(neighbourhood) >= 0 else -1

        all_signs.append(smoothed)
    return all_signs

def segment_arcs(contours, min_arc_length, curvature_window):
    """
    Split a contour at curvature inflection points.
    Each returned arc is a (M,2) float32 array with M >= min_arc_length.

    This is the core of the Lee et al. method: by splitting at inflections
    we obtain arc fragments that each belong to a *single* convex side of
    an ellipse, which makes fitting much more stable.
    """
    all_signs = curvature_signs(contours, window=curvature_window)  # list of sign arrays, one per contour
    arcs = []

    for cnt, signs in zip(contours, all_signs):  # pair each contour with its signs
        pts = cnt.reshape(-1, 2).astype(np.float32)
        start = 0  # reset for every contour

        for i in range(1, len(pts)):
            if signs[i] != signs[i - 1]:
                arc = pts[start:i]
                if len(arc) >= min_arc_length:
                    arcs.append(arc)
                start = i

        # last segment of this contour
        arc = pts[start:]
        if len(arc) >= min_arc_length:
            arcs.append(arc)

    return arcs

def filter_collinear_arcs(arcs, min_curvature):
    """
    Discard arcs that are nearly straight lines by measuring mean curvature magnitude.
    Even if an arc has a consistent curvature sign (passes convexity check), it may
    still be a near-straight line — e.g. a long panel edge — with a very small cross
    product magnitude at every point. These are useless for ellipse fitting.

    The mean cross-product magnitude is computed across all interior points:
        curvature = mean( ||(p[i]-p[i-1]) x (p[i+1]-p[i])|| )
    and normalised by the squared step length to make it scale-invariant.

    Input:
        arcs:            list of (M,2) float32 arrays from segment_arcs
        min_curvature:   minimum mean normalised curvature to keep an arc
                         (tune this: start at 0.1, raise to drop more straight arcs)
    Output:
        filtered:        list of arcs with sufficient curvature
    """
    filtered = []
    for arc in arcs:
        pts = arc.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue

        magnitudes = []
        for i in range(1, len(pts) - 1):
            v1 = pts[i]     - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            cross     = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm      = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6  # avoid div/0
            magnitudes.append(cross / norm)

        mean_curvature = np.mean(magnitudes)
        if mean_curvature >= min_curvature:
            filtered.append(arc)

    return filtered

def menger_radius(arc):
    """Radius of curvature via three representative points on the arc."""
    pts = arc.reshape(-1, 2).astype(np.float64)
    p1, p2, p3 = pts[0], pts[len(pts) // 2], pts[-1]
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    cross = abs((p2 - p1)[0] * (p3 - p1)[1] - (p2 - p1)[1] * (p3 - p1)[0])
    if cross < 1e-6:
        return np.inf  # nearly straight 
    return (a * b * c) / (2 * cross)

def estimate_arc_radius(arc, max_residual_ratio):
    """
    Estimate an arc's radius of curvature via least-squares circle fit
    over ALL points, rather than the 3-point Menger estimate. This is far
    less sensitive to exactly which points happen to be the first/middle/
    last of the arc -- a concern that grows with arc length, since longer
    arcs are exactly the ones carrying the most useful signal (e.g. large
    ring segments) and are also the ones whose 3-point sample is most
    likely to be unrepresentative of the true local curvature.

    Also returns a normalised residual (RMS distance of points from the
    fitted circle, divided by the fitted radius) so that near-straight
    arcs -- which fit an enormous, poorly-constrained "circle" -- can be
    rejected on fit quality rather than relying on radius value alone.

    Input:
        arc:               (M, 2) array of arc points.
        max_residual_ratio: if the fit's normalised RMS residual exceeds
                            this, the arc is considered a poor circular
                            fit (e.g. straight, or noisy) and np.inf is
                            returned regardless of the nominal radius.

    Output:
        radius:            estimated radius (pixels), or np.inf if the
                           arc is too straight / too short / a poor fit.
    """
    pts = arc.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return np.inf

    x, y = pts[:, 0], pts[:, 1]

    # Algebraic circle fit (Kasa method): solve for (cx, cy, r) via
    # linear least squares on x^2+y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf

    cx, cy, c = sol
    r_sq = c + cx**2 + cy**2
    if r_sq <= 0:
        return np.inf
    radius = np.sqrt(r_sq)

    # Residual: how far points actually lie from the fitted circle
    dists = np.sqrt((x - cx)**2 + (y - cy)**2)
    rms_residual = np.sqrt(np.mean((dists - radius) ** 2))
    normalised_residual = rms_residual / radius

    if normalised_residual > max_residual_ratio:
        return np.inf  # poor circular fit -- likely straight or too noisy

    return radius

def filter_arcs_by_radius(arcs, min_radius, max_radius, max_residual_ratio):
    """
    Estimate the radius of curvature of each arc via its menger radius
    and keep the top_n arcs with the largest radii.

    A straight line has an infinite radius of curvature and should be removed first by the collinearity filter. 
    Among the remaining curved arcs, larger radius = gentler curve = more likely to belong to a large ellipse like the ring.

    Input:
        arcs:   list of (M,2) float32 arrays from filter_collinear_arcs
        min_radius:  minimum curvature radius threshold; arcs with radius > min_radius are kept
    Output:
        filtered:  top_n arcs sorted largest radius first
        radii:     corresponding radius values (useful for debugging)
    """
    scored = []
    for arc in arcs:
        r = estimate_arc_radius(arc, max_residual_ratio=max_residual_ratio)
        if r > min_radius and r < max_radius:
            scored.append((arc, r))

    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [arc for arc, _ in scored]
    radii    = [r   for _,   r in scored]

    return filtered, radii

def segment_arcs_by_curvature_smoothness(arcs, max_radius_ratio, min_arc_length):
    """
    Further split arcs where the local radius of curvature changes too abruptly,
    then discard fragments that are too short.

    Motivation: after sign-based segmentation, an arc may still span two different
    ellipses if their boundaries happen to share a convexity direction. A sudden
    jump in curvature magnitude — even without a sign change — betrays this. By
    severing at such jumps we get fragments that are both convex-consistent AND
    curvature-consistent, which are much stronger constraints for ellipse fitting.

    Algorithm:
        1. Compute the local Menger radius at every interior triplet (i-1, i, i+1).
        2. Compute the ratio of consecutive radii: r[i] / r[i-1].
        3. If the ratio exceeds max_radius_ratio (or its inverse, for a drop),
           the arc is severed at point i.
        4. Fragments shorter than min_arc_length are discarded.

    Input:
        arcs:             list of (M, 2) float32 arrays from segment_arcs /
                          filter_collinear_arcs
        max_radius_ratio: maximum allowed ratio between consecutive local radii;
                          a value of 2.0 means the radius may at most double (or
                          halve) between adjacent points before a cut is made.
                          Tighter (e.g. 1.5) = more aggressive splitting.
        min_arc_length:   minimum number of points for a fragment to be kept.

    Output:
        refined:          list of (M, 2) float32 arrays, each smooth in curvature
    """
    refined = []

    for arc in arcs:
        pts = arc.reshape(-1, 2).astype(np.float32)
        n = len(pts)
        if n < 3:
            continue

        # Compute local Menger radius at every interior point
        local_radii = np.full(n, np.inf)
        for i in range(1, n - 1):
            triplet = pts[i-1:i+2]   # shape (3, 2)
            p1, p2, p3 = triplet
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            cross = abs((p2 - p1)[0] * (p3 - p1)[1] - (p2 - p1)[1] * (p3 - p1)[0])
            local_radii[i] = (a * b * c) / (2 * cross) if cross > 1e-6 else np.inf

        # Find cut points where the radius changes too abruptly
        cut_points = []
        for i in range(2, n - 1):
            r_prev = local_radii[i - 1]
            r_curr = local_radii[i]
            # Skip if either is inf (near-straight segment); those are handled
            # by the collinearity filter, not here
            if np.isinf(r_prev) or np.isinf(r_curr):
                continue
            ratio = r_curr / r_prev if r_prev > 0 else np.inf
            if ratio > max_radius_ratio or ratio < 1.0 / max_radius_ratio:
                cut_points.append(i)

        # Sever the arc at cut points and keep long-enough fragments
        boundaries = [0] + cut_points + [n]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            fragment = pts[start:end]
            if len(fragment) >= min_arc_length:
                refined.append(fragment)

    return refined

def compute_target_decimate_density(filtered_arcs, target_total_points=2500,
                                     min_density=0.01, max_density=1.0):
    total_points = sum(len(arc) for arc in filtered_arcs)

    if total_points <= 0:
        print("Warning: total point count is zero, falling back to min_density.")
        return min_density

    density = target_total_points / total_points
    density = float(np.clip(density, min_density, max_density))
    print(f"  Computed decimate density: {density:.4f} "
          f"(total points={total_points}, target={target_total_points}pts)")
    return density


def decimate_arc_uniform_density(arc, points_per_pixel, min_points=5):
    pts = arc.reshape(-1, 2)
    n_pts = len(pts)
    target = int(max(n_pts * points_per_pixel, min_points))
    if n_pts <= target:
        return arc
    idx = np.linspace(0, n_pts - 1, target).astype(int)
    return arc[idx]

######
#Visualisation helpers
######
def visualize(img, title):
    plt.figure(figsize=(12, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"{title}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_ellipse(img, ellipse_candidates, offset=(0,0)):
    x_offset, y_offset = offset

    if img.max() > 255:
        img_8bit = (img / 16).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)

    img_color = cv.cvtColor(img_8bit, cv.COLOR_GRAY2BGR)

    
    for cnt, ellipse, arc_length, eccentricity, density in ellipse_candidates:
        # Draw contour points in red
        pts = cnt.reshape(-1, 2) + np.array([x_offset, y_offset])
        for pt in pts:
            cv.circle(img_color, tuple(pt), 1, (0, 0, 255), -1)
        
        # Shift ellipse center
        (cx, cy), (MA, ma), angle = ellipse
        ellipse_shifted = ((cx + x_offset, cy + y_offset), (MA, ma), angle)

        # Draw fitted ellipse model in green
        cv.ellipse(img_color, ellipse_shifted, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
    plt.title("Detected ellipse model")
    plt.axis('off')
    plt.show()

def visualize_arcs(arcs, img_shape, title):
    canvas = np.zeros(img_shape[:2], dtype=np.uint8)

    for arc in arcs:
        pts = arc.reshape(-1, 2).astype(np.int32)
        for pt in pts:
            if 0 <= pt[1] < img_shape[0] and 0 <= pt[0] < img_shape[1]:
                canvas[pt[1], pt[0]] = 255

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='gray')
    plt.title(f"{title} —  {len(arcs)} arcs")
    plt.axis('off')
    plt.show()


######
#Robust ellipse fit
######
from itertools import combinations

def fit_ellipses_ransac(filtered_arcs, n_ellipses,
                                   min_samples, residual_threshold, max_extra_rounds,
                                   min_delta_sma, max_eccentricity, min_sample_spread, tolerance_eccentricity, tolerance_center,
                                   a_inner_min_px, a_inner_max_px, a_outer_min_px, a_outer_max_px,
                                   samples_per_pair=2, bbox_tracking=None, min_trials = 500, max_trials=1500):
    """
    Iteratively fit ellipses to a pool of arc points using an arc-constrained,
    deterministic RANSAC.

    Rather than drawing a fixed number of random trials, every pair of
    eligible arcs is enumerated exhaustively once per round, with
    `samples_per_pair` five-point samples drawn from each pair. This is
    appropriate for small, heavily pre-filtered arc pools (e.g. 10-50 arcs):
    the number of distinct arc pairs (C(n,2)) is itself small enough that
    exhaustive enumeration guarantees full coverage at a bounded, predictable
    cost, rather than relying on random re-sampling to eventually cover the
    same space.

    Algorithm per round:
        1. For every pair of arcs with >= min_samples points remaining
           (falling back to sampling within a single arc if fewer than 2
           arcs are eligible):
            a. Draw up to `samples_per_pair` five-point samples from the
               pair's pooled points, retrying a bounded number of times per
               sample if the spread check fails.
            b. Fit a candidate EllipseModel to each sample.
            c. Apply cheap rejection gates (tracking bbox, eccentricity,
               size) before scoring.
            d. Score surviving candidates against ALL remaining points;
               track the best.
        2. Refit the best candidate on its full inlier set for a more
           accurate model.
        3. Recompute inliers from the refined model.
        4. Reject if too similar to an already-retained ellipse (by
           semi-major axis).
        5. Remove inliers from the pool (whether retained or not) and repeat.

    Input:
        filtered_arcs:      list of (M, 2) float32/float64 arrays — arc point sets
                            produced by the upstream filtering pipeline.
        n_ellipses:         number of distinct ellipses to extract.
        min_samples:        number of points sampled per RANSAC trial; must be >= 5.
        residual_threshold: inlier distance cutoff in pixels.
        min_delta_sma:      minimum semi-major-axis separation for two ellipses
                            to be considered distinct.
        max_eccentricity:   upper bound on eccentricity before scoring.
        samples_per_pair:   number of five-point samples drawn per arc pair per
                            round (default 2).

    Output:
        retained_ellipses:  list of (xc, yc, a, b, theta, inliers) tuples.
    """

    all_points = np.vstack(filtered_arcs).astype(np.float64)
    print(f"Total points in pool: {len(all_points)}")

    arc_labels = np.concatenate([
        np.full(len(arc), i) for i, arc in enumerate(filtered_arcs)
    ])

    if bbox_tracking is not None:
        bx1, by1, bx2, by2 = bbox_tracking

    retained_ellipses = []
    round_i = 0
    best_overall_ellipse = None
    best_overall_inlier_count = -1

    while round_i < n_ellipses + max_extra_rounds:
        if len(all_points) < min_samples:
            print(f"Round {round_i+1}: not enough points left ({len(all_points)}), stopping.")
            break

        round_i += 1

        unique_arc_ids = np.unique(arc_labels)
        eligible = [aid for aid in unique_arc_ids
                    if np.sum(arc_labels == aid) >= min_samples]

        print(f"RANSAC round {round_i}: {len(all_points)} points remaining, "
              f"{len(eligible)} eligible arcs")

        if len(eligible) == 0:
            print(f"  Round {round_i}: no eligible arcs, stopping.")
            break

        # --- Deterministic pair enumeration ---
        if len(eligible) >= 2:
            arc_pairs = list(combinations(eligible, 2))
        else:
            arc_pairs = [(eligible[0],)]

        rng = np.random.default_rng()

        # --- Standardise trial count: floor at min_trials, ceiling at max_trials ---
        n_full_pairs = len(arc_pairs)
        effective_samples_per_pair = samples_per_pair

        if n_full_pairs * effective_samples_per_pair < min_trials:
            effective_samples_per_pair = int(np.ceil(min_trials / n_full_pairs))
            print(f"  Round {round_i}: only {n_full_pairs} pairs available, increasing "
                  f"samples_per_pair {samples_per_pair} → {effective_samples_per_pair} "
                  f"to reach min_trials={min_trials}")

        max_pairs = max(1, max_trials // effective_samples_per_pair)
        if len(arc_pairs) > max_pairs:
            idx = rng.choice(len(arc_pairs), size=max_pairs, replace=False)
            arc_pairs = [arc_pairs[i] for i in idx]
            print(f"  Round {round_i}: capping {n_full_pairs} eligible pairs down to "
                  f"{max_pairs} (max_trials={max_trials}, samples_per_pair={effective_samples_per_pair})")

        print(f"  Round {round_i}: {len(arc_pairs)} pairs × {effective_samples_per_pair} "
              f"samples/pair = {len(arc_pairs) * effective_samples_per_pair} trials")

        best_model   = None
        best_inliers = None
        best_inlier_count = -1
        n_rejected_bbox = 0
        n_pairs_evaluated = 0

        for pair_ids in arc_pairs:
            sample_pts = np.vstack([all_points[arc_labels == aid] for aid in pair_ids])
            if len(sample_pts) < min_samples:
                continue

            n_pairs_evaluated += 1

            for _ in range(effective_samples_per_pair):
                spread_ok = False
                for _ in range(10):
                    idx    = rng.choice(len(sample_pts), size=min_samples, replace=False)
                    sample = sample_pts[idx]
                    bbox_diagonal = np.linalg.norm(sample.max(axis=0) - sample.min(axis=0))
                    if bbox_diagonal > min_sample_spread:
                        spread_ok = True
                        break
                if not spread_ok:
                    continue

                try:
                    candidate = EllipseModel.from_estimate(sample)
                except (TypeError, ValueError):
                    continue
                if not candidate:
                    continue
                if not np.all(np.isfinite([*candidate.center, *candidate.axis_lengths, candidate.theta])):
                    continue

                if bbox_tracking is not None:
                    cx, cy = candidate.center
                    if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                        n_rejected_bbox += 1
                        continue

                a_c, b_c = candidate.axis_lengths
                semi_major = max(a_c, b_c)
                semi_minor = min(a_c, b_c)
                if semi_major < 1e-6:
                    continue
                ecc = np.sqrt(1.0 - (semi_minor / semi_major) ** 2)
                if ecc > max_eccentricity:
                    continue

                if not (a_inner_min_px < semi_major < a_outer_max_px):
                    continue

                residuals = candidate.residuals(all_points)
                inliers   = np.abs(residuals) < residual_threshold

                if inliers.sum() > best_inlier_count:
                    best_inlier_count = inliers.sum()
                    best_inliers      = inliers
                    best_model        = candidate

        if bbox_tracking is not None:
            print(f"  Round {round_i}: evaluated {n_pairs_evaluated} arc pairs, "
                  f"rejected {n_rejected_bbox} candidates (centre outside bbox)")
        else:
            print(f"  Round {round_i}: evaluated {n_pairs_evaluated} arc pairs")

        if best_model is None:
            print(f"  Round {round_i}: no valid model found, stopping.")
            if len(retained_ellipses) >= 2:
                candidate_pair = select_ellipse_pair(
                    retained_ellipses,
                    min_sma_ratio=sma_ratio_nominal-0.225,
                    max_sma_ratio=min(sma_ratio_nominal+0.075, 0.96),
                    tolerance_center=tolerance_center,
                    tolerance_eccentricity=tolerance_eccentricity,
                    a_outer_min_px=a_outer_min_px,
                    a_outer_max_px=a_outer_max_px
                )
                if candidate_pair is not None:
                    print(f"  Valid pair found after early stop with {len(retained_ellipses)} ellipses.")
                    return retained_ellipses, candidate_pair, None
            break

        # Refit on all inliers for a more accurate final model
        try:
            refined = EllipseModel.from_estimate(all_points[best_inliers])
            if not np.all(np.isfinite([*refined.center, *refined.axis_lengths, refined.theta])):
                raise ValueError("Refit produced non-finite params")
            best_model   = refined
            best_inliers = np.abs(best_model.residuals(all_points)) < residual_threshold
        except (TypeError, ValueError) as e:
            print(f"  Round {round_i}: refit failed ({e}), using best candidate model.")

        xc    = best_model.center[0]
        yc    = best_model.center[1]
        a     = best_model.axis_lengths[0]
        b     = best_model.axis_lengths[1]
        theta = best_model.theta

        current_inlier_count = int(best_inliers.sum())
        if current_inlier_count > best_overall_inlier_count:
            best_overall_inlier_count = current_inlier_count
            best_overall_ellipse = (
                xc, yc, a, b, theta,
                all_points[best_inliers].copy()
            )

        is_duplicate = False
        for (kxc, kyc, ka, kb, ktheta, _) in retained_ellipses:
            if abs(a - ka) < min_delta_sma:
                is_duplicate = True
                break

        if is_duplicate:
            print(f"  Round {round_i}: duplicate, skipping.")
        else:
            inlier_points = all_points[best_inliers].copy()
            retained_ellipses.append((xc, yc, a, b, theta, inlier_points))
            print(f"  → Retained [{len(retained_ellipses)}/{n_ellipses}] "
                  f"center=({xc:.1f}, {yc:.1f})  a={a:.1f}  b={b:.1f}  "
                  f"theta={np.degrees(theta):.1f}°  inliers={best_inliers.sum()}/{len(all_points)}")

            # --- Early-stop check ---
            # As soon as we have >= 2 candidate ellipses, test them against
            # STRICT tolerances (half the normal tolerance_center /
            # tolerance_eccentricity). A pair confident enough to pass this
            # tighter bar lets us skip the remaining rounds entirely, rather
            # than spending extra RANSAC effort hunting for a 3rd ellipse
            # that isn't needed once we already have a solid pair.
            if len(retained_ellipses) >= 2:
                strict_pair = select_ellipse_pair(
                    retained_ellipses,
                    min_sma_ratio=sma_ratio_nominal - 0.225,
                    max_sma_ratio=min(sma_ratio_nominal + 0.075, 0.96),
                    tolerance_center=tolerance_center / 2,
                    tolerance_eccentricity=tolerance_eccentricity / 2,
                    a_outer_min_px=a_outer_min_px,
                    a_outer_max_px=a_outer_max_px
                )
                if strict_pair is not None:
                    print(f"  Early stop: strict pair found after "
                          f"{len(retained_ellipses)} ellipses (round {round_i}), "
                          f"skipping remaining rounds.")
                    return retained_ellipses, strict_pair, None

        all_points  = all_points[~best_inliers]
        arc_labels  = arc_labels[~best_inliers]

        is_last_round = (round_i == n_ellipses + max_extra_rounds)

        if len(retained_ellipses) >= n_ellipses or is_last_round:
            candidate_pair = select_ellipse_pair(
                retained_ellipses,
                min_sma_ratio=sma_ratio_nominal-0.225,
                max_sma_ratio=min(sma_ratio_nominal+0.075, 0.96),
                tolerance_center=tolerance_center,
                tolerance_eccentricity=tolerance_eccentricity,
                a_outer_min_px=a_outer_min_px,
                a_outer_max_px=a_outer_max_px
            )
            if candidate_pair is not None:
                print(f"  Valid pair found after {len(retained_ellipses)} ellipses.")
                return retained_ellipses, candidate_pair, None

    print("Warning: no valid pair found after all rounds.")
    return retained_ellipses, None, best_overall_ellipse

def fit_ellipses_ransac_legacy(filtered_arcs, n_ellipses,
                                   min_samples, residual_threshold, max_trials, max_extra_rounds,
                                   min_delta_sma, max_eccentricity, min_sample_spread, tolerance_eccentricity, tolerance_center,
                                   a_inner_min_px, a_inner_max_px, a_outer_min_px, a_outer_max_px,
                                   bbox_tracking=None):
    """
    Iteratively fit ellipses to a pool of arc points using a arc-constrained RANSAC.

    Unlike standard RANSAC, each trial samples min_samples points from two
    randomly chosen arc rather than the full pool. This enforces geometric consistency
    in the sample — points from one arc belong to one convex side of one ellipse —
    while inlier scoring still runs against all remaining points. After each round,
    inliers are removed so subsequent rounds converge on different ellipses.

    Algorithm per round:
        1. For max_trials iterations:
            a. Pick 2 random arcs with >= min_samples points remaining.
            b. Sample min_samples points from them and fit a candidate EllipseModel.
            c. Reject the candidate if eccentricity > max_eccentricity.
            d. Score the candidate against ALL remaining points; track the best.
        2. Refit the best candidate on its full inlier set for a more accurate model.
        3. Recompute inliers from the refined model.
        4. Reject if too similar to an already-retained ellipse (by semi-major axis).
        5. Remove inliers from the pool (whether retained or not) and repeat.

    Input:
        filtered_arcs:      list of (M, 2) float32/float64 arrays — arc point sets
                            produced by the upstream filtering pipeline.
        n_ellipses:         number of distinct ellipses to extract.
        min_samples:        number of points sampled per RANSAC trial; must be >= 5
                            (the minimum to uniquely determine an ellipse).
        residual_threshold: inlier distance cutoff in pixels — a point is an inlier
                            if |residual| < residual_threshold.
        max_trials:         number of sample-and-score iterations per round.
        min_delta_sma:      minimum difference in semi-major axis length (pixels)
                            for two ellipses to be considered distinct; candidates
                            closer than this to any retained ellipse are discarded.
        max_eccentricity:   upper bound on eccentricity (0 = circle, 1 = parabola);
                            candidates above this are rejected before scoring.
                            Default 0.9.

    Output:
        retained_ellipses:  list of (xc, yc, a, b, theta, inliers) tuples, where
                                xc, yc  — ellipse centre in pixels,
                                a, b    — semi-axis lengths in pixels,
                                theta   — rotation angle in radians,
                                inliers — boolean mask over the points that were in
                                          the pool at the time this ellipse was found.
    """

    all_points = np.vstack(filtered_arcs).astype(np.float64)
    print(f"Total points in pool: {len(all_points)}")

    # Track which arc each point came from, so we can remove inliers arc-aware
    arc_labels = np.concatenate([
        np.full(len(arc), i) for i, arc in enumerate(filtered_arcs)
    ])

    if bbox_tracking is not None:
        bx1, by1, bx2, by2 = bbox_tracking

    retained_ellipses = []
    round_i = 0
    best_overall_ellipse = None
    best_overall_inlier_count = -1

    while round_i < n_ellipses + max_extra_rounds: 
        if len(all_points) < min_samples:
            print(f"Round {round_i+1}: not enough points left ({len(all_points)}), stopping.")
            break

        round_i += 1
        print(f"RANSAC round {round_i}: {len(all_points)} points remaining")

        # --- Manual RANSAC: sample from a single arc, score against all points ---
        best_model   = None
        best_inliers = None
        best_inlier_count = -1

        # Build a lookup of current points per arc (arcs may have shrunk after inlier removal)
        unique_arc_ids = np.unique(arc_labels)

        rng = np.random.default_rng()

        n_rejected_bbox = 0  # diagnostic counter

        for _ in range(max_trials):
            eligible = [aid for aid in unique_arc_ids
                        if np.sum(arc_labels == aid) >= min_samples]
            if not eligible:
                break

            chosen_ids = rng.choice(eligible, size=min(2, len(eligible)), replace=False)
            sample_pts = np.vstack([
                all_points[arc_labels == aid] for aid in chosen_ids
            ])

            # Resample until spread is sufficient or give up after a few attempts
            spread_ok = False
            for _ in range(10):
                idx    = rng.choice(len(sample_pts), size=min_samples, replace=False)
                sample = sample_pts[idx]
                bbox_diagonal = np.linalg.norm(sample.max(axis=0) - sample.min(axis=0))
                if bbox_diagonal > min_sample_spread:
                    spread_ok = True
                    break
            if not spread_ok:
                continue

            # Fit a candidate model to the sample
            try:
                candidate = EllipseModel.from_estimate(sample)
            except (TypeError, ValueError):
                continue
            if not candidate:
                continue
            if not np.all(np.isfinite([*candidate.center, *candidate.axis_lengths, candidate.theta])):
                continue

            # --- Tracking-mode gate: reject centre outside search bbox ---
            # Cheapest possible check (two comparisons), so it goes first,
            # ahead of eccentricity/size checks and well before the O(N)
            # residuals evaluation against the full point pool.
            if bbox_tracking is not None:
                cx, cy = candidate.center
                if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                    n_rejected_bbox += 1
                    continue

            # Eccentricity and SMA bound check on candidate before scoring
            a_c, b_c = candidate.axis_lengths
            semi_major = max(a_c, b_c)
            semi_minor = min(a_c, b_c)
            if semi_major < 1e-6:
                continue
            ecc = np.sqrt(1.0 - (semi_minor / semi_major) ** 2)
            if ecc > max_eccentricity:
                continue

            if not (a_inner_min_px < semi_major < a_outer_max_px):
                continue

            # Score against the FULL point pool
            residuals = candidate.residuals(all_points)
            inliers   = np.abs(residuals) < residual_threshold

            if inliers.sum() > best_inlier_count:
                best_inlier_count = inliers.sum()
                best_inliers      = inliers
                best_model        = candidate

        if bbox_tracking is not None:
                    print(f"  Round {round_i}: rejected {n_rejected_bbox}/{max_trials} candidates (centre outside bbox)")

        if best_model is None:
            print(f"  Round {round_i}: no valid model found, stopping.")
            break

        # Refit on all inliers for a more accurate final model
        try:
            refined = EllipseModel.from_estimate(all_points[best_inliers])
            if not np.all(np.isfinite([*refined.center, *refined.axis_lengths, refined.theta])):
                raise ValueError("Refit produced non-finite params")
            best_model   = refined
            best_inliers = np.abs(best_model.residuals(all_points)) < residual_threshold
        except (TypeError, ValueError) as e:
            print(f"  Round {round_i}: refit failed ({e}), using best candidate model.")

        xc    = best_model.center[0]
        yc    = best_model.center[1]
        a     = best_model.axis_lengths[0]
        b     = best_model.axis_lengths[1]
        theta = best_model.theta

        #Keep track of best ellipse
        current_inlier_count = int(best_inliers.sum())
        if current_inlier_count > best_overall_inlier_count:
            best_overall_inlier_count = current_inlier_count
            best_overall_ellipse = (
                xc,
                yc,
                a,
                b,
                theta,
                all_points[best_inliers].copy()
            )

        # Duplicate check
        is_duplicate = False
        for (kxc, kyc, ka, kb, ktheta, _) in retained_ellipses:
            if abs(a - ka) < min_delta_sma:
                is_duplicate = True
                break

        if is_duplicate:
            print(f"  Round {round_i}: duplicate, skipping.")
        else:
            inlier_points = all_points[best_inliers].copy()
            retained_ellipses.append((xc, yc, a, b, theta, inlier_points))
            print(f"  → Retained [{len(retained_ellipses)}/{n_ellipses}] "
                  f"center=({xc:.1f}, {yc:.1f})  a={a:.1f}  b={b:.1f}  "
                  f"theta={np.degrees(theta):.1f}°  inliers={best_inliers.sum()}/{len(all_points)}")

        # Remove inliers from pool regardless of duplicate status
        all_points  = all_points[~best_inliers]
        arc_labels  = arc_labels[~best_inliers]

        is_last_round = (round_i == n_ellipses + max_extra_rounds)

        if len(retained_ellipses) >= n_ellipses or is_last_round:
            candidate_pair = select_ellipse_pair(
                retained_ellipses,
                min_sma_ratio=sma_ratio_nominal-0.225,   
                max_sma_ratio=min(sma_ratio_nominal+0.075, 0.96),   
                tolerance_center=tolerance_center,
                tolerance_eccentricity=tolerance_eccentricity,
                a_outer_min_px=a_outer_min_px,
                a_outer_max_px=a_outer_max_px
            )
            if candidate_pair is not None:
                print(f"  Valid pair found after {len(retained_ellipses)} ellipses.")
                return retained_ellipses, candidate_pair, None

    print("Warning: no valid pair found after all rounds.")
    return retained_ellipses, None, best_overall_ellipse

def select_ellipse_pair(retained_ellipses, min_sma_ratio, max_sma_ratio, tolerance_center, tolerance_eccentricity, a_outer_min_px, a_outer_max_px):
    """
    From a set of fitted ellipses, select the concentric coplanar pair using
    invariant geometric properties, returning them ordered largest to smallest.

    Motivation: two concentric coplanar circles project to concentric ellipses
    that share invariant properties regardless of camera distance or viewing
    angle: equal eccentricity and coincident centres. The SMA ratio is bounded
    by the known physical geometry. Spurious RANSAC ellipses from other scene
    features are unlikely to satisfy all checks simultaneously.

    Input:
        retained_ellipses:      list of (xc, yc, a, b, theta, inliers) tuples
        min_sma_ratio:          minimum allowed ratio a_inner / a_outer;
                                rejects pairs where the inner ellipse is too small
                                (e.g. thruster paired with outer ring)
        max_sma_ratio:          maximum allowed ratio a_inner / a_outer;
                                should not exceed R_INNER / R_OUTER from CAD
        tolerance_center:       maximum allowed pixel distance between the two
                                ellipse centres; pairs exceeding this are skipped
        tolerance_eccentricity: maximum allowed difference in eccentricity between
                                the two ellipses; pairs exceeding this are skipped

    Output:
        best_pair:              list of two (xc, yc, a, b, theta, inliers) tuples,
                                ordered [outer, inner] (largest to smallest semi-major axis),
                                or None if no pair passed all four filters
    """
    valid_pairs = []

    for i, j in combinations(range(len(retained_ellipses)), 2):
        e_i, e_j = retained_ellipses[i], retained_ellipses[j]

        a_i, a_j = max(e_i[2], e_i[3]), max(e_j[2], e_j[3])
        outer, inner = (e_i, e_j) if a_i > a_j else (e_j, e_i)

        a_outer, b_outer = max(outer[2], outer[3]), min(outer[2], outer[3])
        a_inner, b_inner = max(inner[2], inner[3]), min(inner[2], inner[3])

        # Hard reject 1: SMA ratio outside physically plausible bounds
        sma_ratio = a_inner / a_outer
        if not (min_sma_ratio < sma_ratio < max_sma_ratio):
            continue

        # Hard reject 2: eccentricities too different
        ecc_outer = np.sqrt(1 - (b_outer / a_outer) ** 2)
        ecc_inner = np.sqrt(1 - (b_inner / a_inner) ** 2)
        ecc_diff = abs(ecc_outer - ecc_inner)
        if ecc_diff > tolerance_eccentricity:
            print(f"Rejected pair (ecc): diff={ecc_diff:.3f} > {tolerance_eccentricity}")
            continue

        # Hard reject 3: centres too far apart
        dist_centres = np.sqrt((outer[0] - inner[0])**2 + (outer[1] - inner[1])**2)
        if dist_centres > tolerance_center:
            print(f"Rejected pair (center): dist={dist_centres:.2f} > {tolerance_center}")
            continue

        # Hard reject 4: Outer SMA out of bounds
        if not (a_outer_min_px < a_outer < a_outer_max_px):
            print(f"Rejected pair (outer size): a_outer={a_outer:.1f} outside "
                f"[{a_outer_min_px:.1f}, {a_outer_max_px:.1f}]")
            continue

        valid_pairs.append((ecc_diff, [outer, inner]))

    if len(valid_pairs) == 0:
        print("Warning: no pair passed all four filters, returning None.")
        return None

    #Select pair with smallest eccentricity difference
    best_pair = min(valid_pairs, key=lambda x: x[0])[1]

    if len(valid_pairs) > 1:
        print(f"Info: {len(valid_pairs)} valid pairs found, selecting best by eccentricity.")

    return best_pair


######
#Final best pair refit on full edge map
######
def joint_refit_ellipse_pair(best_pair, edges, x_offset, y_offset, residual_threshold=1.0):
    """
    Refit both ellipses using inliers from the full edge map.

    Algorithm:
        1. Extract all edge pixel coordinates and shift to cropped frame.
        2. For each ellipse, use the existing inlier points to fit an initial
           model, then use it to collect inliers from the full edge map.
        3. Refit via least squares on those inliers.
        Falls back to the original model for any ellipse whose refit fails.

    Input:
        best_pair:          list of two (xc, yc, a, b, theta, inliers) tuples
                            ordered [outer, inner].
        edges:              full-image binary edge map (H x W, uint8), BEFORE cropping.
        x_offset, y_offset: top-left corner of the crop box, used to shift edge
                            points into the cropped coordinate system that the
                            ellipse parameters live in.
        residual_threshold: inlier distance cutoff in pixels; intentionally looser
                            than the RANSAC threshold to maximise coverage (default 1.5).

    Output:
        refined_pair:       list of two (xc, yc, a, b, theta, inliers) tuples with
                            updated parameters; falls back to the original model for
                            any ellipse whose refit fails.
    """
    edge_ys, edge_xs = np.where(edges > 0)
    edge_points = np.column_stack([
        edge_xs - x_offset,
        edge_ys - y_offset,
    ]).astype(np.float64)

    if len(edge_points) < 5:
        print("  joint_refit: not enough edge points, returning original pair.")
        return best_pair

    labels = ["outer", "inner"]
    refined_pair = []

    for label, (xc, yc, a, b, theta, inliers) in zip(labels, best_pair):
        try:
            # Refit from existing inliers to get a clean initial model
            initial_model = EllipseModel.from_estimate(inliers)
            if not np.all(np.isfinite([*initial_model.center, *initial_model.axis_lengths, initial_model.theta])):
                raise ValueError("Non-finite parameters in initial model")

            # Use it to collect inliers from the full edge map
            inlier_mask = np.abs(initial_model.residuals(edge_points)) < residual_threshold
            inlier_pts  = edge_points[inlier_mask]

            print(f"  joint_refit [{label}]: {inlier_mask.sum()} inliers from full edge map")

            if inlier_mask.sum() < 5:
                print(f"  joint_refit [{label}]: too few inliers, keeping original.")
                refined_pair.append((xc, yc, a, b, theta, inlier_pts))
                continue

            # Refit on the full edge map inliers
            refitted = EllipseModel.from_estimate(inlier_pts)
            if not np.all(np.isfinite([*refitted.center, *refitted.axis_lengths, refitted.theta])):
                raise ValueError("Non-finite parameters after refit")

            rxc, ryc = refitted.center
            ra, rb   = refitted.axis_lengths
            rtheta   = refitted.theta

            print(f"  joint_refit [{label}]: "
                  f"center ({xc:.1f},{yc:.1f})→({rxc:.1f},{ryc:.1f})  "
                  f"a {a:.1f}→{ra:.1f}  b {b:.1f}→{rb:.1f}  "
                  f"theta {np.degrees(theta):.1f}°→{np.degrees(rtheta):.1f}°")

            refined_pair.append((rxc, ryc, ra, rb, rtheta, inlier_pts))

        except (TypeError, ValueError) as e:
            print(f"  joint_refit [{label}]: failed ({e}), keeping original.")
            refined_pair.append((xc, yc, a, b, theta, inliers))

    return refined_pair

def absolute_size_filter(fx, R_OUTER, R_INNER, d_min=1.5, d_max=4.0):
    a_outer_max_px = fx * R_OUTER / d_min
    a_outer_min_px = fx * R_OUTER / d_max
    a_inner_max_px = fx * R_INNER / d_min
    a_inner_min_px = fx * R_INNER / d_max
    return a_outer_max_px, a_outer_min_px, a_inner_max_px, a_inner_min_px

def extract_alpha(path):
    match = re.search(r'alpha([+-]?\d+(?:\.\d+)?)', path.stem)
    return float(match.group(1)) if match else None

def extract_scene_number(path):
    match = re.match(r'^scene_(\d+)$', path.stem)
    return int(match.group(1)) if match else None

def angular_coverage_deg(inlier_pts, xc, yc, n_bins=36):
    """Fraction of the ellipse's circumference actually supported by inliers."""
    angles = np.arctan2(inlier_pts[:,1] - yc, inlier_pts[:,0] - xc)
    bins = ((angles + np.pi) / (2*np.pi) * n_bins).astype(int) % n_bins
    occupied = len(np.unique(bins))
    return occupied / n_bins * 360  # degrees of the circle actually "seen"

def normalized_residual(inlier_pts, xc, yc, a):
    dists = np.sqrt((inlier_pts[:,0]-xc)**2 + (inlier_pts[:,1]-yc)**2)
    rms = np.sqrt(np.mean((dists - a)**2))
    return rms / a

def init_tracking_mode(current_file, ellipse_path_dataset, pose_path_dataset, fx,
                        R_TRACKING=0.05, image_size=None, label="outer",
                        frame_prefix="WCAM1_AURICAM"):
    """
    Initialise tracking-mode search parameters from the PREVIOUS frame's
    detection and pose estimate, derived automatically from the current
    frame's filename.

    Given the current frame (e.g. WCAM1_AURICAM_2.png), this extracts the
    frame number (2), decrements it to find the previous frame's number (1),
    and loads that previous frame's saved ellipse fit and pose estimate:
        WCAM1_AURICAM_1_ellipses.npy
        WCAM1_AURICAM_1_ellipses_pose.npy

    Fallback: if frame N-1's files are missing (e.g. that frame was skipped
    due to a failed fit), this falls back ONE further frame to N-2, using a
    doubled R_TRACKING (2 * the value passed in) to compensate for the extra
    frame of potential motion the prior guess is now stale by. If N-2's files
    are also missing, tracking mode is abandoned (returns None) rather than
    walking arbitrarily far back.

    Input:
        current_file:          path (str or Path) to the CURRENT frame being
                               processed, e.g. ".../WCAM1_AURICAM_2.png".
                               Only used to extract the frame number via
                               regex matching `frame_prefix`.
        ellipse_path_dataset:  directory (str or Path) containing
                               *_ellipses.npy files from prior frames.
        pose_path_dataset:     directory (str or Path) containing
                               *_ellipses_pose.npy files from prior frames.
        fx:                    focal length in pixels.
        R_TRACKING:            world-space search radius in metres (default 5cm).
                               Doubled automatically if falling back to N-2.
        image_size:            [width, height] in pixels, used to clip the
                               bounding box to valid image bounds. If None,
                               the box is not clipped.
        label:                 which ellipse to track ("outer" by default).
        frame_prefix:          filename prefix before the frame number, used
                               both to parse the current frame's number and
                               to construct the previous frame's filename.

    Output:
        dict (same as before, plus "frames_back" indicating whether the N-1
        or N-2 frame was used), or None if there is no previous frame (e.g.
        current frame is the first in the sequence) or neither N-1 nor N-2's
        files exist — in which case the caller should fall back to full-frame
        detection (bbox_tracking=None) rather than tracking mode.
    """
    current_file = Path(current_file)

    match = re.search(rf'{frame_prefix}_(\d+)', current_file.stem)
    if match is None:
        print(f"Warning: could not parse frame number from '{current_file.name}' "
              f"using prefix '{frame_prefix}'. Skipping tracking mode.")
        return None

    current_frame_number = int(match.group(1))

    # Try N-1 first, then fall back to N-2 with doubled R_TRACKING if N-1's
    # files are missing.
    candidates = [
        (current_frame_number - 1, R_TRACKING),
        (current_frame_number - 2, R_TRACKING * 2.0),
    ]

    for frames_back, (prev_frame_number, effective_r_tracking) in enumerate(candidates, start=1):
        if prev_frame_number < 1:
            print(f"Info: frame {current_frame_number} has no frame "
                  f"{prev_frame_number} to fall back to. Skipping tracking mode.")
            continue

        ellipse_path = Path(ellipse_path_dataset) / f"{frame_prefix}_{prev_frame_number}_ellipses.npy"
        pose_path    = Path(pose_path_dataset)    / f"{frame_prefix}_{prev_frame_number}_ellipses_pose.npy"

        if not ellipse_path.exists():
            print(f"Warning: previous ellipse file not found ({ellipse_path}).")
            if frames_back == 1:
                print(f"  Falling back to frame {current_frame_number - 2} "
                      f"with R_TRACKING doubled to {R_TRACKING * 2.0:.3f}m.")
                continue
            print("  No further fallback available. Skipping tracking mode.")
            return None

        if not pose_path.exists():
            print(f"Warning: previous pose file not found ({pose_path}).")
            if frames_back == 1:
                print(f"  Falling back to frame {current_frame_number - 2} "
                      f"with R_TRACKING doubled to {R_TRACKING * 2.0:.3f}m.")
                continue
            print("  No further fallback available. Skipping tracking mode.")
            return None

        # Found valid files at this fallback level — use them.
        ellipse_data = np.load(ellipse_path, allow_pickle=True).item()
        ellipse = ellipse_data[label]

        apriori_xc = ellipse["xc"] + ellipse["x_offset"]
        apriori_yc = ellipse["yc"] + ellipse["y_offset"]

        pose_data = np.load(pose_path, allow_pickle=True).item()
        apriori_depth = pose_data[label]["tvec"].item(2)
        center_bounding_circle_px = fx * effective_r_tracking / apriori_depth

        half_size = center_bounding_circle_px
        x1 = apriori_xc - half_size
        y1 = apriori_yc - half_size
        x2 = apriori_xc + half_size
        y2 = apriori_yc + half_size

        if image_size is not None:
            w, h = image_size
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if frames_back == 1:
            print(f"Tracking from frame {prev_frame_number} → {current_frame_number}, "
                  f"bounding box: {(x1, y1, x2, y2)}")
        else:
            print(f"Tracking from frame {prev_frame_number} → {current_frame_number} "
                  f"(fallback, R_TRACKING doubled to {effective_r_tracking:.3f}m), "
                  f"bounding box: {(x1, y1, x2, y2)}")

        return {
            "apriori_xc": apriori_xc,
            "apriori_yc": apriori_yc,
            "apriori_depth": apriori_depth,
            "center_bounding_circle_px": center_bounding_circle_px,
            "bbox": (x1, y1, x2, y2),
            "prev_frame_number": prev_frame_number,
            "frames_back": frames_back,
        }

    print(f"Info: neither frame {current_frame_number - 1} nor "
          f"{current_frame_number - 2} has usable ellipse/pose files. "
          f"Skipping tracking mode.")
    return None

#DEV
# if __name__ == "__main__":
#     if cfg["camera"] == "camera_X":
#         # Camera parameters - CamX
#         focal_length = 6e-3  # m
#         pixel_pitch = 2.2e-6 # m
#         image_size = [2590, 1942]

#         #Parameters based on camera noise
#         max_extra_rounds = 3
#         tolerance_eccentricity = 0.20
#         max_radius_ratio = 3.0

#         fx = focal_length / pixel_pitch
#         fy = focal_length / pixel_pitch

#     elif cfg["camera"] == "AuricamD80":
#         # Camera parameters - AuricamD80
#         image_size = [2048, 2048]
#         # image_size = [1024, 1024] #Binned
#         square_fov_deg = 56.6

#         #Parameters based on camera noise
#         max_extra_rounds = 2
#         tolerance_eccentricity = 0.15
#         max_radius_ratio = 2.0

#         fx = (image_size[0] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)
#         fy = fx

#     #DOR dataset
#     file = r"CPO_dataset_X\1 - DATA\Session_2\shadows\batch_SDW1\run2\raws_tracking\dataset_X_scene_15.png"

#     # --- Load image ---
#     img = read_image(file)

#     # --- Initialise tracking prior EARLY (only needs fx/image_size/prev-frame files) ---
#     bbox_tracking = None
#     tracking = None
#     tracking_ellipse_dir = cfg.get("tracking_ellipse_dir")
#     tracking_pose_dir    = cfg.get("tracking_pose_dir")
#     frame_prefix         = cfg.get("frame_prefix")

#     if tracking_ellipse_dir is not None and tracking_pose_dir is not None:
#         tracking = init_tracking_mode(
#             current_file=file,
#             ellipse_path_dataset=tracking_ellipse_dir,
#             pose_path_dataset=tracking_pose_dir,
#             fx=fx,
#             R_TRACKING=0.05,
#             image_size=image_size,
#             frame_prefix=frame_prefix,
#         )

#     # --- Edge detection + thinning to single-pixel-wide edges ---
#     canny_low = cfg["canny_low"] if DATASET == "DOR" else 100
#     edges = find_edges(img, low=canny_low, high=200) 
#     edges = cv.ximgproc.thinning(edges, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
#     visualize(edges, "Edge map")

#     # --- Depth bounds: tight around apriori depth if tracking, else generic range ---
#     if tracking is not None:
#         depth_margin = 0.5  # metres 
#         d_min = max(cfg["dmin"], tracking["apriori_depth"] - depth_margin)
#         d_max = min(tracking["apriori_depth"] + depth_margin, cfg["dmax"])
#     else:
#         d_min, d_max = cfg["dmin"], cfg["dmax"]

#     #Define min and max sma bounds
#     a_outer_max_px, a_outer_min_px, a_inner_max_px, a_inner_min_px = absolute_size_filter(
#         fx, R_OUTER=cfg["R_OUTER"], R_INNER=cfg["R_INNER"], d_min=d_min, d_max=d_max
#     )

#     # --- Crop to the densest edge region to reduce noise from background ---
#     search_area, (x1, y1, x2, y2) = refine_search_area_symmetric(
#         edges, kernel_size=151, density_threshold=0.05,
#         expected_max_radius_px=a_outer_max_px, margin=1.5*a_outer_max_px
#     )

#     if tracking is not None:
#         x1_bb, y1_bb, x2_bb, y2_bb = tracking["bbox"]
#         bbox_tracking = (
#             x1_bb - x1,
#             y1_bb - y1,
#             x2_bb - x1,
#             y2_bb - y1,
#         )
#         residual_threshold_ransac = 0.5
#     else:
#         residual_threshold_ransac = 0.7

#     # --- Extract contours from the cropped edge map ---
#     contours = find_contours(search_area)

#     # --- Split contours at inflection points to obtain arc fragments ---
#     curvature_window =cfg["curvature_window"] if DATASET == "DOR" else 31
#     arcs = segment_arcs(contours=contours, min_arc_length=int(max(5, 0.8*np.sqrt(a_inner_min_px))), curvature_window=curvature_window)

#     arcs = segment_arcs_by_curvature_smoothness(arcs, max_radius_ratio=max_radius_ratio, min_arc_length=int(max(5, 0.8*np.sqrt(a_inner_min_px))))

#     # --- Drop near-straight arcs ---
#     filtered_arcs_collinear = filter_collinear_arcs(arcs, min_curvature=0.2)

#     # --- Keep only the top-N largest-radius arcs ---
#     min_radius = 0.98*a_inner_min_px
#     max_radius = 1.67*a_outer_max_px
#     filtered_arcs_radius, radii = filter_arcs_by_radius(filtered_arcs_collinear, min_radius=min_radius, max_radius=max_radius, max_residual_ratio=0.3)

#     print("Number of contours: ", len(contours))
#     print("Number of arcs fragmentation, filtering by length and convexity: ", len(arcs))
#     print("Number of arcs after filtering by collinearity", len(filtered_arcs_collinear))
#     print("Number of arcs after filtering by Menger radius", len(filtered_arcs_radius))

#     print(f"\n{'='*70}\nArc radius diagnostic ({len(filtered_arcs_collinear)} arcs)\n{'='*70}")
#     for i, arc in enumerate(filtered_arcs_collinear):
#         pts = arc.reshape(-1, 2)
#         chord_len = np.linalg.norm(pts[-1] - pts[0])
#         n_pts = len(pts)

#         r_menger = menger_radius(arc)

#         x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
#         if n_pts >= 5:
#             A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
#             b = x**2 + y**2
#             sol, *_ = np.linalg.lstsq(A, b, rcond=None)
#             cx, cy, c = sol
#             r_sq = c + cx**2 + cy**2
#             if r_sq > 0:
#                 r_fit = np.sqrt(r_sq)
#                 dists = np.sqrt((x - cx)**2 + (y - cy)**2)
#                 rms_residual = np.sqrt(np.mean((dists - r_fit) ** 2))
#                 residual_ratio = rms_residual / r_fit
#             else:
#                 r_fit, residual_ratio = np.inf, np.inf
#         else:
#             r_fit, residual_ratio = np.inf, np.inf

#         print(f"  arc[{i:3d}]  n_pts={n_pts:4d}  chord_len={chord_len:7.1f}  "
#             f"menger_r={r_menger:10.2f}  fit_r={r_fit:8.2f}  residual_ratio={residual_ratio:.3f}")

#     # --- Arc pipeline visualisation  ---
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     titles = [
#         f"Contours ({len(contours)})",
#         f"Arcs after segmentation ({len(arcs)})",
#         f"Arcs after radius filter ({len(filtered_arcs_radius)})",
#     ]

#     for ax, stage, title in zip(axes, [contours, arcs, filtered_arcs_radius], titles):
#         canvas = np.zeros(search_area.shape[:2], dtype=np.uint8)
#         for item in stage:
#             pts = np.array(item).reshape(-1, 2).astype(np.int32)
#             for pt in pts:
#                 if 0 <= pt[1] < search_area.shape[0] and 0 <= pt[0] < search_area.shape[1]:
#                     canvas[pt[1], pt[0]] = 255
#         ax.imshow(canvas, cmap='gray')
#         ax.set_title(title)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

#     #Fit 3 distinct ellipses with remaining point pool using RANSAC, removing all inlier points between successive ellipse fits
#     decimate_density = compute_target_decimate_density(filtered_arcs_radius, target_total_points=3000)
#     filtered_arcs_for_ransac = [
#         decimate_arc_uniform_density(arc, points_per_pixel=decimate_density, min_points=5)
#         for arc in filtered_arcs_radius]

#     tolerance_center = min(a_outer_max_px / 7.5, 80)
#     min_delta_sma = max(10.0, 0.8*(a_outer_min_px-a_inner_min_px))

#     retained_ellipses, best_pair, best_overall_ellipse = fit_ellipses_ransac(
#         filtered_arcs=filtered_arcs_for_ransac, n_ellipses=3, min_samples=5,
#         residual_threshold=residual_threshold_ransac, min_delta_sma=min_delta_sma,
#         max_eccentricity=0.8, min_sample_spread=1.6*a_inner_min_px, max_extra_rounds=max_extra_rounds,
#         tolerance_eccentricity=tolerance_eccentricity,
#         tolerance_center=tolerance_center,
#         a_inner_min_px=a_inner_min_px, a_inner_max_px=a_inner_max_px,
#         a_outer_min_px=a_outer_min_px, a_outer_max_px=a_outer_max_px,
#         bbox_tracking=bbox_tracking
#     )

#     # Visualise ellipses on edge map
#     if search_area.max() > 255:
#         display = (search_area / 16).astype(np.uint8)
#     else:
#         display = search_area.astype(np.uint8)
#     display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

#     n = len(retained_ellipses)
#     mpl_colors = [cm.tab10(i) for i in range(n)]
#     cv_colors  = [tuple(int(c * 255) for c in mpl_colors[i][:3])[::-1] for i in range(n)]
#     labels     = ["ellipse_1", "ellipse_2"] + [f"ellipse_{i}" for i in range(n - 2)]

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
#     for i, (xc, yc, a, b, theta, inliers) in enumerate(retained_ellipses[:3]):
#         cv.ellipse(display_color, (int(round(xc)), int(round(yc))),
#                 (int(round(a)), int(round(b))), float(np.degrees(theta)),
#                 0, 360, cv_colors[i], 2)
#         ax.annotate(labels[i], xy=(xc, yc), xytext=(xc + a + 10, yc),
#                     color=mpl_colors[i], fontsize=12, fontweight='bold')
#     ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
#     ax.set_title(f"Detected ellipses — edge map")
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

#     # Visualise ellipses on original image
#     img_crop = img[y1:y2, x1:x2]
#     if img_crop.max() > 255:
#         display = (img_crop / 16).astype(np.uint8)
#     else:
#         display = img_crop.astype(np.uint8)
#     display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     for i, (xc, yc, a, b, theta, inliers) in enumerate(retained_ellipses):
#         cv.ellipse(display_color, (int(round(xc)), int(round(yc))),
#                 (int(round(a)), int(round(b))), float(np.degrees(theta)),
#                 0, 360, cv_colors[i], 2)
#         ax.annotate(labels[i], xy=(xc, yc), xytext=(xc + a + 10, yc),
#                     color=mpl_colors[i], fontsize=12, fontweight='bold')
#     ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
#     ax.set_title("Detected ellipses — original image")
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

#     if best_pair is None:
#         print("Warning: skipping frame, no valid pair found")
#         mpl_colors = ['lime']
#         cv_colors   = [(0, 255, 0)]
#         labels      = ["best ellipse"]
#         img_crop = img[y1:y2, x1:x2]
#         if img_crop.max() > 255:
#             display = (img_crop / 16).astype(np.uint8)
#         else:
#             display = img_crop.astype(np.uint8)
#         display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

#         fig, ax = plt.subplots(figsize=(10, 10))
#         xc, yc, a, b, theta, inliers = best_overall_ellipse
#         cv.ellipse(display_color, (int(round(xc)), int(round(yc))),
#                 (int(round(a)), int(round(b))), float(np.degrees(theta)),
#                 0, 360, cv_colors[0], 2)
#         ax.annotate(labels[0], xy=(xc, yc), xytext=(xc + a + 10, yc),
#                     color=mpl_colors[0], fontsize=12, fontweight='bold')
#         ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
#         ax.set_title("Refitted ellipses — original image")
#         ax.axis('off')
#         plt.tight_layout()
#         plt.show()
#     else:
#         refitted_pair = joint_refit_ellipse_pair(best_pair=best_pair, edges=edges, x_offset=int(x1), y_offset=int(y1), residual_threshold=1.0)
#         refitted_pair = joint_refit_ellipse_pair(best_pair=refitted_pair, edges=edges, x_offset=int(x1), y_offset=int(y1), residual_threshold=0.3)

#         mpl_colors = ['lime', 'orange']
#         cv_colors   = [(0, 255, 0), (0, 165, 255)]
#         labels      = ["outer", "inner"]
#         img_crop = img[y1:y2, x1:x2]
#         if img_crop.max() > 255:
#             display = (img_crop / 16).astype(np.uint8)
#         else:
#             display = img_crop.astype(np.uint8)
#         display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

#         fig, ax = plt.subplots(figsize=(10, 10))
#         for i, (xc, yc, a, b, theta, inliers) in enumerate(refitted_pair):
#             cv.ellipse(display_color, (int(round(xc)), int(round(yc))),
#                     (int(round(a)), int(round(b))), float(np.degrees(theta)),
#                     0, 360, cv_colors[i], 2)
#             ax.annotate(labels[i], xy=(xc, yc), xytext=(xc + a + 10, yc),
#                         color=mpl_colors[i], fontsize=12, fontweight='bold')
#         ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
#         ax.set_title("Refitted ellipses — original image")
#         ax.axis('off')
#         plt.tight_layout()
#         plt.show()

#FULL DATASET
if __name__ == "__main__":
    if cfg["camera"] == "camera_X":
        # Camera parameters - CamX
        focal_length = 6e-3  # m
        pixel_pitch = 2.2e-6 # m
        image_size = [2590, 1942]

        #Parameters tuned based on camera characteristics
        max_extra_rounds = 3
        tolerance_eccentricity = 0.20
        max_radius_ratio = 3.0

        fx = focal_length / pixel_pitch
        fy = focal_length / pixel_pitch

    elif cfg["camera"] == "AuricamD80":
        # Camera parameters - AuricamD80
        image_size = [2048, 2048]
        # image_size = [1024, 1024] #Binned
        square_fov_deg = 56.6

        #Parameters tuned based on camera characteristics
        max_extra_rounds = 2
        tolerance_eccentricity = 0.15
        max_radius_ratio = 2.0

        fx = (image_size[0] / 2.0) / np.tan(np.deg2rad(square_fov_deg) / 2.0)
        fy = fx

    dataset_dir = cfg["dataset_dir"]
    output_dir  = cfg["output_dir"]
    output_dir.mkdir(exist_ok=True)   # parents=True since results/updated_model_DOR_23072026/ may not exist yet

    image_extensions = {".tiff", ".tif", ".png"}
    if cfg.get("frame_prefix") is not None:
        files = [
            f for f in dataset_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        pattern = re.compile(rf'^{re.escape(cfg["frame_prefix"])}_(\d+)$')

        files = [f for f in files if pattern.match(f.stem)]

        files.sort(
            key=lambda f: int(pattern.match(f.stem).group(1))
        )

        files = [f for f in files if int(pattern.match(f.stem).group(1)) > 79]

    else:
        files = [
            f for f in dataset_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

    # files = [f for f in files if (a := extract_alpha(f)) is not None and a >= 70] #Only run for alpha>=70
    # files = [f for f in files if (s := extract_scene_number(f)) is not None and s > 79]  # Only run for scene > x
    # files = [f for f in files if (m := re.search(r'WCAM1_AURICAM_(\d+)', f.stem)) and int(m.group(1)) > 124 and int(m.group(1)) < 300]

    if not files:
        print(f"No image files found in {dataset_dir}")

    mpl_colors = ["lime", "orange"]
    cv_colors  = [(0, 255, 0), (0, 165, 255)]
    labels     = ["outer", "inner"]

    K = np.array([
        [fx, 0,  image_size[0] / 2],
        [0,  fx, image_size[1] / 2],
        [0,  0,   1]
    ], dtype=np.float64)

    pose_output_dir = Path(str(output_dir) + "_pose")
    pose_output_dir.mkdir(exist_ok=True)

    for file in files:
        print(f"\n{'='*60}\nProcessing: {file.name}\n{'='*60}")
        stem = file.stem  # filename without extension, used for all outputs

        try:
            # --- Load image ---
            img = read_image(file)

            # --- Initialise tracking prior EARLY (only needs fx/image_size/prev-frame files) ---
            bbox_tracking = None
            tracking = None
            tracking_ellipse_dir = cfg.get("tracking_ellipse_dir")
            tracking_pose_dir    = cfg.get("tracking_pose_dir")
            frame_prefix         = cfg.get("frame_prefix")

            if tracking_ellipse_dir is not None and tracking_pose_dir is not None:
                tracking = init_tracking_mode(
                    current_file=file,
                    ellipse_path_dataset=tracking_ellipse_dir,
                    pose_path_dataset=tracking_pose_dir,
                    fx=fx,
                    R_TRACKING=0.05,
                    image_size=image_size,
                    frame_prefix=frame_prefix,
                )

            # --- Edge detection + thinning ---
            canny_low = cfg["canny_low"] if DATASET == "DOR" else 100
            edges = find_edges(img, low=canny_low, high=200)
            edges = cv.ximgproc.thinning(edges, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

            # --- Depth bounds: tight around apriori depth if tracking, else generic range ---
            if tracking is not None:
                depth_margin = 0.5  # metres 
                d_min = max(cfg["dmin"], tracking["apriori_depth"] - depth_margin)
                d_max = min(tracking["apriori_depth"] + depth_margin, cfg["dmax"])
            else:
                d_min, d_max = cfg["dmin"], cfg["dmax"]

            a_outer_max_px, a_outer_min_px, a_inner_max_px, a_inner_min_px = absolute_size_filter(
                fx, R_OUTER=cfg["R_OUTER"], R_INNER=cfg["R_INNER"], d_min=d_min, d_max=d_max
            )

            # --- Crop to the densest edge region to reduce noise from background ---
            # search_area, (x1, y1, x2, y2) = refine_search_area(
            #     edges, kernel_size=81, density_threshold=0.05, margin=20
            # )
            # search_area, (x1, y1, x2, y2) = refine_search_area_symmetric(edges, kernel_size=151, density_threshold=0.05, expected_max_radius_px=a_outer_max_px, margin=150)
            search_area = edges
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]

            # search_area, (x1, y1, x2, y2) = refine_search_area_symmetric(
            #     edges, kernel_size=151, density_threshold=0.05,
            #     expected_max_radius_px=a_outer_max_px, margin=1.5*a_outer_max_px
            # )

            if tracking is not None:
                x1_bb, y1_bb, x2_bb, y2_bb = tracking["bbox"]
                bbox_tracking = (
                    x1_bb - x1,
                    y1_bb - y1,
                    x2_bb - x1,
                    y2_bb - y1,
                )
                residual_threshold_ransac = 0.5
            else:
                residual_threshold_ransac = 0.7

            # --- Extract contours from the cropped edge map ---
            contours = find_contours(search_area)

            # --- Split contours at inflection points to obtain arc fragments ---
            curvature_window =cfg["curvature_window"] if DATASET == "DOR" else 31
            arcs = segment_arcs(contours=contours, min_arc_length=int(max(5, 0.8*np.sqrt(a_inner_min_px))), curvature_window=curvature_window)

            arcs = segment_arcs_by_curvature_smoothness(arcs, max_radius_ratio=max_radius_ratio, min_arc_length=int(max(5, 0.8*np.sqrt(a_inner_min_px))))

            # --- Drop near-straight arcs ---
            filtered_arcs_collinear = filter_collinear_arcs(arcs, min_curvature=cfg["min_curvature"])

            # --- Keep only the top-N largest-radius arcs ---
            min_radius = 0.98*a_inner_min_px
            max_radius = 1.67*a_outer_max_px
            filtered_arcs_radius, radii = filter_arcs_by_radius(filtered_arcs_collinear, min_radius=min_radius, max_radius=max_radius, max_residual_ratio=0.3)
            print("Min radius: ", min_radius)
            print("Max radius: ", max_radius)

            print("Number of contours:", len(contours))
            print("Arcs after segmentation + convexity filter:", len(arcs))
            print("Arcs after collinearity filter:", len(filtered_arcs_collinear))
            print("Arcs after Menger radius filter:", len(filtered_arcs_radius))

            if not filtered_arcs_radius:
                print(f"  Skipping {file.name}: no arcs survived filtering.")
                continue

            # --- Fit ellipses ---
            decimate_density = compute_target_decimate_density(filtered_arcs_radius, target_total_points=3000)
            filtered_arcs_for_ransac = [
                decimate_arc_uniform_density(arc, points_per_pixel=decimate_density, min_points=5)
                for arc in filtered_arcs_radius]           
            max_ransac_restarts = 2
            retained_ellipses, best_pair, best_overall_ellipse = None, None, None

            #Define tolerance center wrt a_outer_max_px
            tolerance_center = min(a_outer_max_px / 7.5, 80)
            min_delta_sma = max(10.0, 0.8*(a_outer_min_px-a_inner_min_px))

            for attempt in range(max_ransac_restarts):
                retained_ellipses, best_pair, best_overall_ellipse = fit_ellipses_ransac(
                    filtered_arcs=filtered_arcs_for_ransac, n_ellipses=3, min_samples=5,
                    residual_threshold=residual_threshold_ransac, min_delta_sma=min_delta_sma, max_eccentricity=0.8,
                    min_sample_spread=1.6*a_inner_min_px, max_extra_rounds=max_extra_rounds,
                    tolerance_eccentricity=tolerance_eccentricity, tolerance_center=tolerance_center,
                    a_inner_min_px=a_inner_min_px, a_inner_max_px=a_inner_max_px,
                    a_outer_min_px=a_outer_min_px, a_outer_max_px=a_outer_max_px,
                    bbox_tracking=bbox_tracking, min_trials=600, max_trials=1000
                )
                if best_pair is not None:
                    break
                print(f"  Attempt {attempt+1}/{max_ransac_restarts}: no valid pair, retrying...")

            if not retained_ellipses:
                print(f"  Skipping {file.name}: RANSAC found no ellipses.")
                continue

            if best_pair is None:
                print(f"  Skipping {file.name}: no valid ellipse pair found after {max_ransac_restarts} attempts.")
                continue
            else:
                refitted_pair = joint_refit_ellipse_pair(best_pair=best_pair, edges=edges, x_offset=int(x1), y_offset=int(y1), residual_threshold=2.0)
                refitted_pair = joint_refit_ellipse_pair(best_pair=refitted_pair, edges=edges, x_offset=int(x1), y_offset=int(y1), residual_threshold=0.5)

                # --- Reject if outer/inner centers disagree too much after refit ---
                (xc_outer, yc_outer, *_rest_outer) = refitted_pair[0]
                (xc_inner, yc_inner, *_rest_inner) = refitted_pair[1]
                center_dist = np.sqrt((xc_outer - xc_inner)**2 + (yc_outer - yc_inner)**2)

                if center_dist > tolerance_center / 3:
                    print(f"  Skipping {file.name}: post-refit center_dist={center_dist:.2f}px > threshold.")
                    continue

                # --- Save ellipse parameters ---
                # Each ellipse stored as dict: xc, yc, a, b, theta_deg
                ellipse_data = {}
                for label, (xc, yc, a, b, theta, inlier_pts) in zip(labels, refitted_pair):
                    # Fit-quality metrics: how well-supported is this individual
                    # ellipse's size/shape, independent of the other ellipse in the pair
                    if len(inlier_pts) >= 5:
                        coverage_deg = angular_coverage_deg(inlier_pts, xc, yc, n_bins=36)
                        resid_norm   = normalized_residual(inlier_pts, xc, yc, a)
                    else:
                        coverage_deg = 0.0
                        resid_norm   = np.inf

                    ellipse_data[label] = {
                        "xc":       float(xc),
                        "yc":       float(yc),
                        "a":        float(a),
                        "b":        float(b),
                        "theta_deg": float(np.degrees(theta)),
                        "inliers": inlier_pts.astype(np.float32),
                        "n_inliers": int(len(inlier_pts)),
                        "angular_coverage_deg": float(coverage_deg),
                        "normalized_residual": float(resid_norm),
                        # offsets to map back to full-image coordinates
                        "x_offset": int(x1),
                        "y_offset": int(y1),
                    }
                np.save(output_dir / f"{stem}_ellipses.npy", ellipse_data)

                # --- Compute and save pose for this frame ---
                # Required so init_tracking_mode can find this frame's pose
                # when processing the NEXT frame in the sequence.
                try:
                    pose_data = compute_pose(
                        ellipse_data, K,
                        R_OUTER = cfg["R_OUTER"], R_INNER = cfg["R_INNER"]
                    )
                    np.save(pose_output_dir / f"{stem}_ellipses_pose.npy", pose_data)
                except Exception as e:
                    print(f"  WARNING: pose estimation failed for {file.name}: {e}. "
                          f"Tracking mode will fall back to full-frame search on the next frame.")

                # --- Visualise ellipses on cropped original image ---
                img_crop = img[y1:y2, x1:x2]
                display  = (img_crop / 16).astype(np.uint8) if img_crop.max() > 255 \
                        else img_crop.astype(np.uint8)
                display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

                fig, ax = plt.subplots(figsize=(10, 10))
                for i, (xc, yc, a, b, theta, _inliers) in enumerate(refitted_pair):
                    cv.ellipse(
                        display_color,
                        (int(round(xc)), int(round(yc))),
                        (int(round(max(a, b))), int(round(min(a, b)))),
                        float(np.degrees(theta)),
                        0, 360,
                        cv_colors[i], 2,
                    )
                    ax.annotate(
                        labels[i], xy=(xc, yc),
                        xytext=(xc + max(a, b) + 10, yc),
                        color=mpl_colors[i], fontsize=12, fontweight="bold",
                    )
                ax.imshow(cv.cvtColor(display_color, cv.COLOR_BGR2RGB))
                ax.set_title(f"Detected ellipses — {file.name}")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(output_dir / f"{stem}_ellipses.png", dpi=150)
                plt.close(fig)  # free memory; essential when looping over many files

        except Exception as e:
            print(f"  ERROR processing {file.name}: {e}")
            continue
