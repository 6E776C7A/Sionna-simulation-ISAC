from filterpy.kalman import KalmanFilter
import numpy as np

def initialize_kalman_filter(dt, process_variance, measurement_variance):
    kf = KalmanFilter(dim_x=6, dim_z=3)

    # State transition matrix
    kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
    # Measurement function
    kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]])

    # Process noise covariance
    kf.Q = process_variance * np.eye(6)

    # Measurement noise covariance
    kf.R = measurement_variance * np.eye(3)

    # Initial state estimate
    kf.x = np.zeros((6, 1))

    # Initial covariance estimate
    kf.P = np.eye(6) * 100

    kf.is_initialized = False

    return kf

def kalman_tracking(kf, range_m, azimuth_deg, elevation_deg):

    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.radians(elevation_deg)

    x_meas = range_m * np.cos(el_rad) * np.cos(az_rad)
    y_meas = range_m * np.cos(el_rad) * np.sin(az_rad)
    z_meas = range_m * np.sin(el_rad)

    z = np.array([[x_meas], [y_meas], [z_meas]])

    if not kf.is_initialized:
        kf.x[:3] = np.array([[x_meas], [y_meas], [z_meas]])
        kf.x[3:6] = np.array([[0], [0], [0]])
        kf.is_initialized = True
    else:
        kf.predict()
        kf.update(z)

    kalman_x = round(float(kf.x[0, 0]), 6)
    kalman_y = round(float(kf.x[1, 0]), 6)
    kalman_z = round(float(kf.x[2, 0]), 6)
    kalman_vx = round(float(kf.x[3, 0]), 6)
    kalman_vy = round(float(kf.x[4, 0]), 6)
    kalman_vz = round(float(kf.x[5, 0]), 6)

    kalman_velocity_m_s = round(float(np.sqrt(kalman_vx**2 + kalman_vy**2 + kalman_vz**2)), 6)

    kalman_result = {
        "kalman_x": kalman_x,
        "kalman_y": kalman_y,
        "kalman_z": kalman_z,
        "kalman_vx": kalman_vx,
        "kalman_vy": kalman_vy,
        "kalman_vz": kalman_vz,
        "kalman_velocity_m_s": kalman_velocity_m_s,
    }           

    return kalman_result 