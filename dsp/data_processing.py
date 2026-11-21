import numpy as np
import ofdm_config as cfg

def load_and_pad_h_cfr(filename):
    """
    Load the simulation data from a .npz file and pad the h_cfr array to match the target size.

    Parameters:
    - filename: Path to the .npz file containing the simulation data.

    Returns:
    - h_cfr_padded: Padded channel frequency response data.
    """

    try:
        with np.load(filename) as data:
            if 'h_cfr' in data:
                h_cfr = data['h_cfr']
            else:
                raise KeyError(f"'h_cfr' not found in {filename}. Please check the file contents.")
        print("Data loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found. Please check the path and try again.")

    if h_cfr.shape[-2] > cfg.num_symbols:
        h_cfr = h_cfr[..., :cfg.num_symbols, :]

    if h_cfr.shape[-2] < cfg.num_symbols:
        sym_pad = cfg.doppler_fft_size - h_cfr.shape[-2]
        pad_widht = [(0, 0)] * (h_cfr.ndim - 2) + [(0, sym_pad), (0, 0)]
        h_cfr = np.pad(h_cfr, pad_widht, mode='constant', constant_values=0)

    target_size = cfg.fft_size
    current_size = h_cfr.shape[-1]
    if current_size < target_size:
        padding_size = target_size - current_size
        padding_before = padding_size // 2
        padding_after = padding_size - padding_before

        pad_width = [(0, 0)] * (h_cfr.ndim - 1) + [(padding_before, padding_after)]
        h_cfr_padded = np.pad(h_cfr, pad_width, mode='constant', constant_values=0)
    else:
        print(f"No padding needed. Current size ({current_size}) is greater than or equal to target size ({target_size}).")
        h_cfr_padded = h_cfr

    return h_cfr_padded

def save_processed_data(proper_frame_id, frame_time, detections, music_result=None):
    """
    Save the processed data (detections and MUSIC results) to a .npz file.

    Parameters:
    - filename: Path to the output .npz file.
    - detections: Array of detected targets.
    - music_result: Dictionary containing MUSIC results (azimuth and elevation).
    """

    proper_time_s = round(proper_frame_id * frame_time, 1)

    if detections:

        strongest_detection = max(detections, key=lambda detection: detection['power'])

        delay_idx = strongest_detection['delay_idx']
        doppler_idx = strongest_detection['doppler_idx']

        range_m = cfg.c * cfg.delay_bins[delay_idx] * 1e-9 / 2
        velocity_m_s = cfg.wavelength * cfg.doppler_bins[doppler_idx] / 2
        velocity_m_s = -velocity_m_s  # Adjusting the sign to match the expected direction


        return {
            "proper_frame_id": int(proper_frame_id),
            "proper_time_s": proper_time_s,
            "detected": True,
            "delay_idx": int(delay_idx),
            "doppler_idx": int(doppler_idx),
            "range_m": round(float(range_m), 6),
            "velocity_m_s": round(float(velocity_m_s), 6),
            "music_azimuth_deg": round(float(music_result["azimuth_deg"]), 6),
            "music_elevation_deg": round(float(music_result["elevation_deg"]), 6),
            "power": f"{float(strongest_detection['power']):.6e}",
        }
    
    return {
        "proper_frame_id": int(proper_frame_id),
        "proper_time_s": proper_time_s,
        "detected": False,
        "delay_idx": None,
        "doppler_idx": None,
        "range_m": None,
        "velocity_m_s": None,
        "music_azimuth_deg": None,
        "music_elevation_deg": None,
        "power": None,
    }

def save_reference_data_to_file(reference_data_list, filename):
    """
    Save a list of reference data dictionaries to a CSV file.

    Parameters:
    - reference_data_list: A list of dictionaries containing reference data.
    - filename: The name of the CSV file to save the data to.

    Returns:
    None
    """
    import csv

    if not reference_data_list:
        print("No reference data to save.")
        return

    # Get the keys from the first dictionary as headers
    headers = reference_data_list[0].keys()

    try:
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for data in reference_data_list:
                writer.writerow(data)
        print(f"Reference data saved to {filename}.")
    except Exception as e:
        print(f"An error occurred while saving reference data: {e}")