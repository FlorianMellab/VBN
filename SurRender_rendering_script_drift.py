from surrender.surrender_client import surrender_client
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import tifffile
from scipy.stats import qmc
from scipy.spatial.transform import Rotation

#IMPORTANT: the true angles are saved to the metadata with the convention roll=about x, pitch=about y, yaw=about z.
#In pose estimation script, these angles are converted to the standard aeronautical convention such that
#roll = about z, pitch=about x and yaw = about y

#--[CONSTANTS]---------------------------
SUN_RADIUS         =    696342000.0 #m
EARTH_RADIUS       =      6478137.0 #m
EARTH_SUN_DISTANCE = 149597870000.0 #m

"""
This function performs the multiplication of two quaternions <q1>x<q2>
Parameters:
q1,q2 : the quaternion to multiply
"""
def quatMultiplication(q1,q2):
  x1=q1[0];    x2=q2[0]
  y1=q1[1];    y2=q2[1]
  z1=q1[2];    z2=q2[2]
  w1=q1[3];    w2=q2[3]

  x = w2*x1 + x2*w1 + y2*z1 - z2*y1
  y = w2*y1 - x2*z1 + y2*w1 + z2*x1
  z = w2*z1 + x2*y1 - y2*x1 + z2*w1
  w = w2*w1 - x2*x1 - y2*y1 - z2*z1

  return np.array([x,y,z,w])

def wrap180(angle_deg):
    return (angle_deg + 180) % 360 - 180

def main(s: surrender_client):
    parameters = {}

    #--[Connection to server]--------------------------------
    s.setVerbosityLevel(1)
    s.connectToServer("127.0.0.1", 5151)

    print("----------------------------------------")
    print("SCRIPT : %s"%sys.argv[0])
    print("SurRender version: "+s.version())
    print("----------------------------------------")

    #--[Initialisation]--------------------------
    s.closeViewer()
    s.setConventions(s.XYZ_SCALAR_CONVENTION,s.Z_FRONTWARD)
    s.enableDoublePrecisionMode( True )
    s.enableRaytracing( True )

    #--[Objects creation]---------------------
    # --[Satellite]-------------------------

    # Create BRDFs - Source from NASA optical properties
    # Gold MLI — bright, slightly redder
    s.createBRDF("mli_brdf", "phong.brdf", {
    })

    s.createBRDF("mli_sides_brdf", "phong.brdf", {
    })

    # Solar arrays — highly absorptive, slight NIR reflectance
    s.createBRDF("solar_brdf", "mate.brdf", {
        "albedo": [0.12, 0.10, 0.10, 0.12]
    })

    # Metallic structure (aluminium) — flat, high reflectance
    s.createBRDF("metal_brdf", "mate.brdf", {
        "albedo": [0.92, 0.90, 0.90, 0.90]
    })

    # Nozzle (oxidised metal/carbon) — dark, low reflectance
    s.createBRDF("nozzle_brdf", "mate.brdf", {
        "albedo": [0.12, 0.12, 0.12, 0.15]
    })

    parameters.update({
        "mli_albedo": [1.0, 1.0, 1.0, 1.0],
        "SA_albedo": [0.12, 0.10, 0.10, 0.12],
        "structure_albedo": [0.92, 0.90, 0.90, 0.90],
        "nozzle_albedo": [0.12, 0.12, 0.12, 0.15]
    })

    # Load mesh — model in meters
    s.createMesh('sat', 'geo_sat_scaled_origin.obj', 1.0)
    s.createMesh('rise', 'rise_blender_panels_rotated.obj', 1.0)

    # Assign per element — element name must match group name in .obj
    s.setObjectElementBRDF('sat', 'mli_Mesh', 'mli_brdf')
    s.setObjectElementBRDF('sat', 'mli_sides_Mesh', 'mli_sides_brdf')
    s.setObjectElementBRDF('sat', 'structure_Mesh', 'metal_brdf')
    s.setObjectElementBRDF('sat', 'solar_array_Mesh',  'solar_brdf')
    s.setObjectElementBRDF('sat', 'adapter_ring_Mesh', 'metal_brdf')
    s.setObjectElementBRDF('sat', 'nozzle_Mesh', 'nozzle_brdf')

    # Assign BRDF to the mesh
    s.setObjectElementBRDF('rise', 'rise', 'metal_brdf')

    # Satellite position
#    satOriginZ = 0.049675 #Offset to get 2m distace to outer ring
#    satOriginY = -0.06193
    satOriginZ = -0.22105 #Offset measured from blender to get 2m distance to outer ring
    # (CAD world origin flush with panel)
    satOriginY = 0.0
    xSatPos = 0.0
    ySatPos = 0.0
    zSatPos = -2.0
    xRisePosNominal = 0.0
    yRisePosNominal = 0.0
    zRisePos = 1.70
    s.setObjectPosition("sat", (xSatPos, ySatPos + satOriginY, zSatPos + satOriginZ))
    s.setObjectPosition('rise', (xRisePosNominal, yRisePosNominal, zRisePos))

    parameters.update({
        "sat_position" : [xSatPos, ySatPos, zSatPos],
        "rise_position_nominal" : [xRisePosNominal, yRisePosNominal, zRisePos]
    })

    # Satellite attitude
    u = np.array([0, 1, 0])
    angle = np.pi
    axis = u/np.linalg.norm(u) * np.sin(angle/2)
    quaternion_sat = np.array( axis.tolist() + [np.cos(angle/2)])

    # Rotate 90° clockwise around Z
    u = np.array([0, 0, 1])
    angle = -np.pi/2  # negative = clockwise (right-hand rule about +Z)
    axis = u / np.linalg.norm(u) * np.sin(angle/2)
    quat2 = np.array(axis.tolist() + [np.cos(angle/2)])

    quaternion_rise_nominal = quatMultiplication(quaternion_sat, quat2)
    s.setObjectAttitude("sat", quaternion_sat)
    s.setObjectAttitude('rise', quaternion_rise_nominal)

    parameters.update({
        "sat_attitude" : [quaternion_sat.tolist()],
        "rise_attitude_nominal" : [quaternion_rise_nominal.tolist()]
    })
    R_baseline_rise = Rotation.from_quat(quaternion_rise_nominal)

    #--[Objects creation]---------------------
    # Earth
    s.createBRDF("mate", "mate.brdf", {})
    s.createShape("earth_shape", "sphere.shp", {'radius': EARTH_RADIUS})
    s.createBody("earth", "earth_shape", "mate", ["earth.jpg"])

    # Earth position — behind camera at GEO distance
    xEarthPos = 0
    yEarthPos = 0
    zEarthPos = -(EARTH_RADIUS + 35786e3)
    s.setObjectPosition("earth", (xEarthPos, yEarthPos, zEarthPos))

    # Rotate -66.5° around Y (Summer solstice pointing)
    u = np.array([0, 1, 0])
    angle = -66.5/180*np.pi
    axis = u / np.linalg.norm(u) * np.sin(angle/2)
    quat1 = np.array(axis.tolist() + [np.cos(angle/2)])

    # Rotate -70° around Z (European longitude)
    u = np.array([0, 0, 1])
    angle = -70.0/180 * np.pi
    axis = u / np.linalg.norm(u) * np.sin(angle/2)
    quat2 = np.array(axis.tolist() + [np.cos(angle/2)])

    # Rotate -23.5 around X (North pointing up)
    u = np.array([1,0,0])
    angle = -23.5/180*np.pi
    axis  = u / np.linalg.norm(u) * np.sin(angle/2)
    quat3 = np.array(axis.tolist() + [np.cos(angle/2)])

    #Earth attitude
    quat4 = quatMultiplication(quat1, quat2)
    quaternion = quatMultiplication(quat3, quat4)
    s.setObjectAttitude("earth", quaternion)

    # Sun
    s.createBRDF("sun",    "sun.brdf",    {})
    s.createShape("sun_shape", "sphere.shp", {'radius':SUN_RADIUS})
    s.createBody("sun", "sun_shape", "sun", [])

    #--[Camera position]-----------------------
    xCamPosNominal = 0.66
    yCamPosNominal = 0.23
    zCamPos = 0.0
    s.setObjectPosition( "camera", ( xCamPosNominal, yCamPosNominal, zCamPos ) )

    # Nominal camera attitude
#    angle = 0.0
#    u = np.array([0, 0, 1])
#    axis = u / np.linalg.norm(u) * np.sin(angle / 2)
#    quaternion_camera_nominal = np.array(axis.tolist() + [np.cos(angle / 2)])
#    s.setObjectAttitude("camera", quaternion_camera_nominal)
    q_default = s.getObjectAttitude("camera")
    print("Default cam attitude", q_default)
    R_baseline_camera = Rotation.from_quat(q_default)

    #--[Image size (WAC)]------------------------
    imWidth  = 2048
    imHeight = 2048
    s.setImageSize(imWidth, imHeight)

    #--[FOV configuration (WAC)]------------------------
    xFOV = 56.6 #deg
    yFOV = 56.6 #deg
    s.setCameraFOVDeg(xFOV,yFOV)

    #--[PSF and ray noise]-----------------

    s.loadPSFModel('gaussian.psf', {
        'sigma': 0.6,
        'epsilon': 1e-2,
        'sigma_radius': 3
    })
    psfSigma = 0.6
    nbSamples = 64
    s.enableRegularPSFSampling(True)
    s.setNbSamplesPerPixel(nbSamples)

    #--[Sensor model (AURICAM WAC)]------------------------
    s.runLuaScript('sensors/GenericSensor.lua')

    #Define WAC model
    s.runLuaCode('GenericSensor.config.lens_surface = 3.14157*(1.25e-3)^2')
    s.runLuaCode('GenericSensor.config.lens_transmittance=0.75')
    s.runLuaCode('GenericSensor.config.bandwidth = 1.6e7')
    s.runLuaCode('GenericSensor.config.shutter="global"')
    s.runLuaCode('GenericSensor.config.line_addressing_time=4e-6')
    s.runLuaCode('GenericSensor.config.QE = vec4(0.52, 0.60, 0.48, 0.12)')
    s.runLuaCode('GenericSensor.config.spectrum = {vec4(400,500,580,750)*1e-9, vec4(500,580,750,1000)*1e-9}')
    s.runLuaCode('GenericSensor.config.fill_factor = 0.98')
    s.runLuaCode('GenericSensor.config.readout_noise = 18')
    s.runLuaCode('GenericSensor.config.ADC_gain = 0.011')
    s.runLuaCode('GenericSensor.config.DC = 70')
    s.runLuaCode('GenericSensor.config.DC_NU = 7.5')
    s.runLuaCode('GenericSensor.config.PRNU = 1.5')
    s.runLuaCode('GenericSensor.config.Nb_lines = 2048')
    s.runLuaCode('GenericSensor.config.Nb_cols = 2048')
    s.runLuaCode('GenericSensor.config.integration_time = 0.003')
    s.runLuaCode('GenericSensor.config.depth = 12')

    #Setup
    s.runLuaCode('GenericSensor:setup()')

    parameters.update({
        "cam_position"          : [xCamPosNominal, yCamPosNominal, zCamPos],
        "image_width"           : imWidth,
        "image_height"          : imHeight,
        "fov_x_deg"             : xFOV,
        "fov_y_deg"             : yFOV,
        "psf_sigma"             : psfSigma,
        "nb_samples_per_pixel"  : nbSamples,
        "sensor_lens_surface"  : 3.14157 * (1.25e-3)**2,
        "sensor_transmittance" : 0.75,
        "sensor_bandwidth"     : 1.6e7,
        "sensor_shutter"       : "global",
        "sensor_QE"            : [0.52, 0.60, 0.45, 0.12],
        "sensor_fill_factor"   : 0.98,
        "sensor_readout_noise" : 18,
        "sensor_ADC_gain"      : 0.011,
        "sensor_DC"            : 70,
        "sensor_DC_NU"         : 7.5,
        "sensor_PRNU"          : 1.5,
        "sensor_Nb_lines"      : 2048,
        "sensor_Nb_cols"       : 2048,
        "sensor_integration_time": 0.003,
        "sensor_depth"         : 12,
    })

    # Sensitivity study parameter space
    n_samples = 500

    # --- Attitude + drift (Latin Hypercube Sampling) ---
    sampler = qmc.LatinHypercube(d=5, seed=42)
    sample = sampler.random(n=n_samples)

    l_bounds = [-8, -8, -8, -0.60, -0.40]
    u_bounds = [8,  8,  8,  0.20,  0.40]
    scaled = qmc.scale(sample, l_bounds, u_bounds)

    roll, pitch, yaw     = scaled[:,0], scaled[:,1], scaled[:,2]
    x_drift, y_drift     = scaled[:,3], scaled[:,4]

    rng = np.random.default_rng(42)  # match whatever seed you use elsewhere for reproducibility

    # sun_alpha: uniform over [-80,-40] U [40,80], skipping [-40,40]
    alpha_low_high  = np.array([-70, -40])
    alpha_high_low  = np.array([40, 70])
    use_positive = rng.random(n_samples) < 0.5  # 50/50 which side of the gap
    sun_alpha = np.where(
        use_positive,
        rng.uniform(alpha_high_low[0], alpha_high_low[1], n_samples),
        rng.uniform(alpha_low_high[0], alpha_low_high[1], n_samples),
    )

    # sun_beta: uniform over [-10, 10]
    sun_beta = rng.uniform(-10, 10, n_samples)

    # --- Combined dataset ---
    dataset = list(zip(roll, pitch, yaw, x_drift, y_drift, sun_alpha, sun_beta))

    #Save dataset
    output_dir = "CPO_dataset_sensitivity_panels_rotated_scaled"
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(metadata_path):
        # Resume: reuse existing file (already has ground truth for processed scenes)
        with open(metadata_path, "r") as f:
            parameters = json.load(f)
    else:
        # First run: build edge_case_dataset from scratch
        parameters["edge_case_dataset"] = [
            {
                "scene": i,
                "roll": float(r),
                "pitch": float(p),
                "yaw": float(y),
                "x_drift": float(x),
                "y_drift": float(yd),
                "sun_alpha": float(alpha),
                "sun_beta": float(beta)
            }
            for i, (r, p, y, x, yd, alpha, beta) in enumerate(dataset)
        ]
        with open(metadata_path, "w") as f:
            json.dump(parameters, f, indent=4)

    # Precompute once, outside the loop
    pivot     = np.array([xRisePosNominal, yRisePosNominal, zRisePos])
    cam_offset = np.array([xCamPosNominal, yCamPosNominal, zCamPos]) - pivot
    print("Camera lever arm: ", cam_offset)

    # Fixed ring feature pose in world coordinates (SurRender origin frame)
    P_ring_world = np.array([0.0, 0.0, -2.0])
    R_ring_world = Rotation.from_quat(quaternion_sat)   # ring rotates rigidly with sat's mesh/body frame

    ground_truth_records = []
    for i, (r, p, y, x, yd, alpha, beta) in enumerate(dataset):

        filename = f"CPO_dataset_sensitivity_panels_rotated_scaled/scene_{i:04d}.tiff"
        entry = parameters["edge_case_dataset"][i]

        # Skip scenes that are already fully done (ground truth computed + image rendered)
        if "x_true" in entry and os.path.exists(filename):
            continue

        R_drift = Rotation.from_euler('xyz', [r, p, y], degrees=True)

        quaternion_rise   = (R_drift * R_baseline_rise).as_quat()
        quaternion_camera = (R_drift * R_baseline_camera).as_quat()
        s.setObjectAttitude("camera", quaternion_camera)
        s.setObjectAttitude("rise", quaternion_rise)

        translation   = np.array([x, yd, 0.0])
        rise_new_pos  = pivot + translation
        cam_new_pos   = rise_new_pos + R_drift.apply(cam_offset)

        s.setObjectPosition("camera", tuple(cam_new_pos))
        s.setObjectPosition("rise", tuple(rise_new_pos))

        # --- Exact ground truth: ring feature pose wrt camera, in camera's own frame ---
        R_cam_world = Rotation.from_quat(quaternion_camera)

        vec_world  = P_ring_world - cam_new_pos
        t_true_cam = R_cam_world.inv().apply(vec_world)
        print("x: ", t_true_cam[0], "y: ", t_true_cam[1], "z: ", t_true_cam[2])

        # Fixed correction: the pose estimator's rotation convention is rotated
        # -90 deg about the camera's own boresight (z) relative to SurRender's
        # convention. Confirmed via cross-correlation: roll_est correlated with
        # -pitch_true, pitch_est correlated with +roll_true, before this fix.
        AXIS_CORRECTION = Rotation.from_euler('z', -90, degrees=True)
        R_true_cam = R_cam_world.inv() * R_ring_world * AXIS_CORRECTION
        roll_true, pitch_true, yaw_true = R_true_cam.as_euler('xyz', degrees=True)

        print("Roll: ", roll_true, "Pitch: ", pitch_true, "yaw_true: ", wrap180(yaw_true-90))
        ground_truth_records.append({
            "scene": i,
            "x_true": float(t_true_cam[0]),
            "y_true": float(t_true_cam[1]),
            "z_true": float(t_true_cam[2]),
            "roll_true": float(roll_true),
            "pitch_true": float(pitch_true),
            "yaw_true": float(wrap180(yaw_true-90)),
        })

        # Merge ground truth into this scene's metadata entry immediately
        parameters["edge_case_dataset"][i].update({
            "x_true": float(t_true_cam[0]),
            "y_true": float(t_true_cam[1]),
            "z_true": float(t_true_cam[2]),
            "roll_true": float(roll_true),
            "pitch_true": float(pitch_true),
            "yaw_true": float(wrap180(yaw_true-90)),
        })
#    for i, (r, p, y, x, yd, alpha, beta) in enumerate(dataset):
#        R_drift = Rotation.from_euler('xyz', [r, p, y], degrees=True)
#
#        # Attitudes (unchanged — already correct)
#        quaternion_rise   = (R_drift * R_baseline_rise).as_quat()
#        quaternion_camera = (R_drift * R_baseline_camera).as_quat()
#        s.setObjectAttitude("camera", quaternion_camera)
#        s.setObjectAttitude("rise", quaternion_rise)
#
#        # Rigid-body position: translate pivot, then rotate camera's lever arm about it
#        translation   = np.array([x, yd, 0.0])
#        rise_new_pos  = pivot + translation
#        cam_new_pos   = rise_new_pos + R_drift.apply(cam_offset)
#
#        s.setObjectPosition("camera", tuple(cam_new_pos))
#        s.setObjectPosition("rise", tuple(rise_new_pos))
#
        # Set sun position
        xSunPos = EARTH_SUN_DISTANCE * np.sin(np.deg2rad(alpha))
        ySunPos = EARTH_SUN_DISTANCE * np.cos(np.deg2rad(alpha)) * np.sin(np.deg2rad(beta))
        zSunPos = EARTH_SUN_DISTANCE * np.cos(np.deg2rad(alpha)) * np.cos(np.deg2rad(beta))

        s.setObjectPosition("sun", (xSunPos, ySunPos, zSunPos))

        s.runLuaCode('GenericSensor:render()')

        image = s.getImage()
        image_12bit = (image * (2**12 - 1)).astype(np.uint16)

        tifffile.imwrite(filename, image_12bit)

        print(f"Sun angles — alpha: {alpha:.2f}°, beta: {beta:.2f}°")
        print("----------------------------------------")
        print("max:", image_12bit.max())
        print("mean:", image_12bit.mean())
        print("shape:", image_12bit.shape)

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(parameters, f, indent=4)

    print("Satellite scene done!")

if __name__ == "__main__":
    s = surrender_client()
    main(s)





















