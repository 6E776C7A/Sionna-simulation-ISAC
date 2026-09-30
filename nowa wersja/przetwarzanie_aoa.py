import numpy as np
import matplotlib.pyplot as plt
import ofdm_config as cfg

taps = np.load("historia_taps.npy", allow_pickle=True)

num_angle_bins = 128
angle_bins_deg = np.arcsin(np.linspace(-1, 1, num_angle_bins)) * (180 / np.pi)  # Convert to degrees

tap_indices = np.arange(-6, 101)
delay_bins = tap_indices * cfg.delay_resolution * 1e9  # Convert to nanoseconds

for steps in range(len(taps)):
    a, tau = taps[steps]

    h_step = a[0, :, 0, 0, :, :]
    print(f"Krok {steps}: kształt h_step: {h_step.shape}")

    h_upa = h_step.reshape(cfg.rx_rows, cfg.rx_cols, cfg.num_symbols, -1)

    h_clean = h_upa - np.mean(h_upa, axis=2, keepdims=True)

    window_sym = np.hanning(cfg.num_symbols)[None,None, :, None]
    h_doppler = np.fft.fft(h_clean * window_sym, axis=2, norm="ortho")
    h_doppler_shifted = np.fft.fftshift(h_doppler, axes=2)

    window_angle = np.hanning(cfg.rx_cols)[None, :, None, None]
    h_azimuth = np.fft.fft(h_doppler_shifted * window_angle, n=num_angle_bins, axis=1)
    h_azimuth_shifted = np.fft.fftshift(h_azimuth, axes=1)

    map_angle_range = np.max(np.abs(h_azimuth_shifted)**2, axis=(0, 2))
    map_db = 10 * np.log10(map_angle_range + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.imshow(map_db, aspect='auto', extent=[delay_bins[0], delay_bins[-1], angle_bins_deg[0], angle_bins_deg[-1]], origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Angle (degrees)')
    plt.title(f'Angle-Delay Map - Step {steps}')
    plt.show()