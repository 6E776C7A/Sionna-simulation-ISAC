import data_processing as dp
import mti_filter as mti
import delay_doppler as dd
import ofdm_config as cfg
import cfar as cfar
import music as music
import numpy as np
import kalman_filter as kf

if __name__ == "__main__":

    # Load and pad data from the .npz file
    #filename = "./wyniki/default_16_256/Human/simulation_data_human_1.npz"
    filename = "./wyniki/default_16_256/Car_D/simulation_data_car_D_1.npz"
    h_cfr, h_cfr_padded = dp.load_and_pad_h_cfr(filename)
    
    antena_number = int(np.sqrt(h_cfr_padded.shape[2]))

    music_radius = int(np.ceil(antena_number / 2))

    antena_positions = np.load(
        f"./wyniki/antena_configs/"
        f"antena_positions_{antena_number}x{antena_number}.npy"
    )

    #filter_methods = ['dc_mean', 'dc_recursive', 'butter', 'cheby2']
    mti_filter_params = mti.get_mti_filter_params('cheby2', cutoff_hz=100.0)

    frame_id = 10

    frame_time = 0.1

    kalman_filter = kf.initialize_kalman_filter(
        dt=frame_time,
        process_variance=0.05,
        measurement_variance=1.0
    )

    # Delay-Doppler processing
    h_delay_doppler_all, delay_doppler_map = (
        dd.compute_delay_doppler_map(
            h_cfr_padded,
            snapshot_id=frame_id,
            **mti_filter_params,
        )
    )

    # Visualize the results of the delay-Doppler map
    # dd.visualize_3d_delay_doppler_map(delay_doppler_map, max_delay_ns=2000, max_doppler_hz=6000)

    dd.visualize_magnitude_map(delay_doppler_map)

    dd.visualize_power_map(delay_doppler_map)

    # CFAR processing
    detection_map, threshold_map, noise_map = cfar.compute_ca_cfar_map(delay_doppler_map)

    cfar.visualize_ca_cfar_map(detection_map, threshold_map, noise_map)

    detections = cfar.extract_detected_targets(
        delay_doppler_map,
        detection_map,
        threshold_map,
        exclude_zero_doppler=False,
    )

    cfar.visualize_clustered_detections(detections, delay_doppler_map)

    music_result = None
    
    if detections:
    
        strongest_detection = max(
            detections,
            key=lambda detection: detection["power"],
        )

        delay_idx = strongest_detection["delay_idx"]
        doppler_idx = strongest_detection["doppler_idx"]

        delay_ns = cfg.delay_bins[delay_idx]
        doppler_hz = cfg.doppler_bins[doppler_idx]

        delay_s = delay_ns * 1e-9

        range_m = cfg.c * delay_s / 2
        velocity_m_s = cfg.wavelength * doppler_hz / 2

        print(f"Delay: {delay_ns:.2f} ns")
        print(f"Doppler: {doppler_hz:.2f} Hz")
        print(f"Range: {range_m:.2f} m")
        print(f"Radial velocity: {velocity_m_s:.2f} m/s")

        music_result = music.compute_music(
            h_delay_doppler_all,
            doppler_idx=doppler_idx,
            delay_idx=delay_idx,
            num_sources=1,
            doppler_radius=music_radius,
            delay_radius=music_radius,
            antena_positions=antena_positions,
            azimuth_grid_deg=np.linspace(-90.0, 90.0, 361),
            elevation_grid_deg=np.linspace(-90.0, 90.0, 361)
        )

        print(f"MUSIC azimuth: {music_result['azimuth_deg']:.2f} deg")
        print(f"MUSIC elevation: {music_result['elevation_deg']:.2f} deg")

        music.visualize_music_spectrum(music_result)
