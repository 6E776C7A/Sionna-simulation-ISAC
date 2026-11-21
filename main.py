
import glob
import os
import gc
import dsp.data_processing as dp
import dsp.mti_filter as mti
import dsp.delay_doppler as dd
import dsp.cfar as cfar
import dsp.music as music
import numpy as np
import dsp.kalman_filter as kf

if __name__ == "__main__":

    # Load and pad data from the .npz file
    folder_path = "./wyniki/default_16_256/Drone/"
    npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
    npz_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))

    basename = "_".join(os.path.basename(npz_files[0]).split("_")[:-1])
    output_filename = os.path.join(folder_path, f"{basename}_results.csv")

    #filter_methods = ['dc_mean', 'dc_recursive', 'butter', 'cheby2']
    mti_filter_params = mti.get_mti_filter_params('cheby2', cutoff_hz=100.0)

    frame_time = 0.1

    kalman_filter = kf.initialize_kalman_filter(
        dt=frame_time,
        process_variance=0.05,
        measurement_variance=1.0
    )

    results = []

    for filename in npz_files:
        print(f"Processing file: {filename}")
    
        h_cfr_padded = dp.load_and_pad_h_cfr(filename)

        print(f"Loaded data from {filename}")

        chunk_number = int(os.path.basename(filename).split("_")[-1].split(".")[0])
        chunk_size = h_cfr_padded.shape[0]

        antena_number = int(np.sqrt(h_cfr_padded.shape[2]))
        antena_positions = np.load(
            f"./wyniki/antena_configs/"
            f"antena_positions_{antena_number}x{antena_number}.npy"
        )

        music_radius = int(np.ceil(antena_number / 2))

        for i in range(chunk_size):

            frame_id = i

            proper_frame_id = (chunk_number - 1) * chunk_size + frame_id + 1

            proper_time_s = proper_frame_id * frame_time

            # Delay-Doppler processing
            h_delay_doppler_all, delay_doppler_map = (
                dd.compute_delay_doppler_map(
                    h_cfr_padded,
                    snapshot_id=frame_id,
                    **mti_filter_params,
                )
            )

            # CFAR processing
            detection_map, threshold_map, noise_map = cfar.compute_ca_cfar_map(delay_doppler_map)

            # Extract detected targets
            detections = cfar.extract_detected_targets(
                delay_doppler_map,
                detection_map,
                threshold_map,
                exclude_zero_doppler=True,
            )

            music_result = None
            
            if detections:
            
                strongest_detection = max(
                    detections,
                    key=lambda detection: detection["power"],
                )

                delay_idx = strongest_detection["delay_idx"]
                doppler_idx = strongest_detection["doppler_idx"]

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

            result = dp.save_processed_data(
                proper_frame_id=proper_frame_id,
                frame_time=frame_time,
                detections=detections,
                music_result=music_result if detections else None,
            )

            # Apply Kalman filter to the results
            if detections:
                kalman_result = kf.kalman_tracking(
                    kalman_filter, range_m=result["range_m"], 
                    azimuth_deg=result["music_azimuth_deg"], 
                    elevation_deg=result["music_elevation_deg"]
                )

                result.update(kalman_result)
            else:
                empty_kalman = {
                    "kalman_x": None,
                    "kalman_y": None,
                    "kalman_z": None,
                    "kalman_vx": None,
                    "kalman_vy": None,
                    "kalman_vz": None,
                    "kalman_velocity_m_s": None,
                }
                result.update(empty_kalman)

            results.append(result)

        print(f"Finished processing file: {filename}")
        del h_cfr_padded
        gc.collect()

    # Save the results to csv file
    dp.save_reference_data_to_file(results, output_filename)
