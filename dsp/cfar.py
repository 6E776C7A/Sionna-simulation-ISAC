import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import ofdm_config as cfg

def compute_ca_cfar_map(delay_doppler_map):
    """
    Compute the CFAR (Constant False Alarm Rate) map from the delay-Doppler map.

    Parameters:
    - delay_doppler_map: 2D numpy array representing the delay-Doppler map.
    Returns:
    - detection_map: 2D boolean numpy array indicating detected targets.
    - threshold_map: 2D numpy array representing the threshold values for each cell.
    - noise_map: 2D numpy array representing the estimated noise power for each cell.
    """

    guard_doppler = 4
    guard_delay = 3
    train_doppler = 8
    train_delay = 10
    p_fa = 1e-6  # Desired probability of false alarm

    power_map = np.abs(delay_doppler_map) ** 2

    doppler_radius = guard_doppler + train_doppler
    delay_radius = guard_delay + train_delay

    window_size_doppler = 2 * doppler_radius + 1
    window_size_delay = 2 * delay_radius + 1

    guard_size_doppler = 2 * guard_doppler + 1
    guard_size_delay = 2 * guard_delay + 1

    num_train_cells = window_size_doppler * window_size_delay - guard_size_doppler * guard_size_delay

    alpha = num_train_cells * (p_fa ** (-1 / num_train_cells) - 1)

    training_mask = np.ones((window_size_doppler, window_size_delay), dtype=bool)

    center_doppler = doppler_radius
    center_delay = delay_radius

    training_mask[center_doppler - guard_doppler:center_doppler + guard_doppler + 1,
                  center_delay - guard_delay:center_delay + guard_delay + 1] = False

    assert np.sum(training_mask) == num_train_cells, "Training mask does not have the correct number of training cells."

    detection_map = np.zeros(power_map.shape, dtype=bool)
    threshold_map = np.full(power_map.shape, np.nan)
    noise_map = np.full(power_map.shape, np.nan)

    for i in range(doppler_radius, power_map.shape[0] - doppler_radius):
        for j in range(delay_radius, power_map.shape[1] - delay_radius):
            window = power_map[i - doppler_radius:i + doppler_radius + 1,
                                       j - delay_radius:j + delay_radius + 1]

            training_cells = window[training_mask]

            noise_estimate = np.mean(training_cells)
            threshold = alpha * noise_estimate

            threshold_map[i, j] = threshold
            noise_map[i, j] = noise_estimate

            if power_map[i, j] > threshold:
                detection_map[i, j] = True

    return detection_map, threshold_map, noise_map

def extract_detected_targets(
    delay_doppler_map,
    detection_map,
    threshold_map,
    exclude_zero_doppler=False,
):
    """
    Extract the detected targets' delay and Doppler indices from the detection map.

    Parameters:
    - delay_doppler_map: 2D numpy array representing the delay-Doppler map.
    - detection_map: 2D boolean numpy array indicating detected targets.
    - threshold_map: 2D numpy array representing the threshold values for each cell.
    - exclude_zero_doppler: If True, reject detections in the central
      Doppler bin. Use only when stationary targets are not part of the scenario.
    Returns:
    - detections: List of dictionaries containing the detected targets' information.
    """

    power_map = np.abs(delay_doppler_map) ** 2

    structure = np.ones((3, 3), dtype=bool)

    labels, num_labels = ndimage.label(detection_map, structure=structure)

    detections = []

    for label_id in range(1, num_labels + 1):
        doppler_indices, delay_indices = np.nonzero(labels == label_id)

        if len(doppler_indices) == 0 or len(delay_indices) == 0:
            continue

        group_power = power_map[doppler_indices, delay_indices]

        strongest_idx = np.argmax(group_power)

        doppler_idx = doppler_indices[strongest_idx]
        delay_idx = delay_indices[strongest_idx]

        zero_doppler_idx = delay_doppler_map.shape[0] // 2
        if exclude_zero_doppler and doppler_idx == zero_doppler_idx:
            continue

        detections.append({
            'doppler_idx': doppler_idx,
            'delay_idx': delay_idx,
            'power': group_power[strongest_idx],
            'threshold': threshold_map[doppler_idx, delay_idx]
        })

    return detections

def visualize_ca_cfar_map(detection_map, threshold_map, noise_map):
    """
    Visualize the CFAR detection map, threshold map, and noise map.

    Parameters:
    - detection_map: 2D boolean numpy array indicating detected targets.
    - threshold_map: 2D numpy array representing the threshold values for each cell.
    - noise_map: 2D numpy array representing the estimated noise power for each cell.
    Returns:
    - None
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    extent = [cfg.delay_bins[0], cfg.delay_bins[-1], cfg.doppler_bins[0], cfg.doppler_bins[-1]]
    xlabel, ylabel = 'Delay (ns)', 'Doppler Shift (Hz)'

    im0 = axs[0].imshow(detection_map, cmap='gray', aspect='auto', origin='lower', extent=extent)
    axs[0].set_title('CFAR Detection Map')
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(threshold_map, cmap='viridis', aspect='auto', origin='lower', extent=extent)
    axs[1].set_title('CFAR Threshold Map')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(noise_map, cmap='plasma', aspect='auto', origin='lower', extent=extent)
    axs[2].set_title('Estimated Noise Map')
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel(ylabel)
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlim(0, 1500)
        ax.set_ylim(-10000, 10000)

    plt.tight_layout()
    plt.show()

def visualize_clustered_detections(detections, delay_doppler_map):
    """
    Visualize the clustered detections on the delay-Doppler map.

    Parameters:
    - detections: List of dictionaries containing the detected targets' information.
    - delay_doppler_map: 2D numpy array representing the delay-Doppler map.
    Returns:
    - None
    """

    plt.figure(figsize=(10, 6))

    power_map_db = 10 * np.log10(np.abs(delay_doppler_map)**2 + 1e-12)

    delays = cfg.delay_bins
    dopplers = cfg.doppler_bins

    mesh = plt.pcolormesh(delays, dopplers, power_map_db, cmap='viridis', shading='nearest')

    plt.title('Delay-Doppler Map with Clustered Detections')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Doppler Shift (Hz)')
    plt.colorbar(mesh, label='Magnitude (dB)')

    label_moving = False
    label_static = False

    for detection in detections:
        doppler_idx = detection['doppler_idx']
        delay_idx = detection['delay_idx']

        target_delay = delays[delay_idx]
        target_doppler = dopplers[doppler_idx]

        if target_doppler < 1.0:
            label = 'Static Target' if not label_static else None
            plt.plot(target_delay, target_doppler, 'bx', markersize=10, markeredgewidth=2, label=label)
            label_static = True
        else:
            label = 'Moving Target' if not label_moving else None
            plt.plot(target_delay, target_doppler, 'rx', markersize=10, markeredgewidth=2, label=label)
            label_moving = True

    plt.xlim(0, 1500)
    plt.ylim(-6000, 6000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()