import numpy as np
import matplotlib.pyplot as plt
import ofdm_config as cfg

taps = np.load("historia_taps_dron.npy", allow_pickle=True)

doppler_bins = np.arange(-cfg.num_symbols // 2, cfg.num_symbols // 2) * cfg.doppler_resolution

for steps in range(len(taps)):
    a, tau = taps[steps]

    h_step = a[0, :, 0, 0, :, :]
    print (f"Krok {steps}: kształt h_step: {h_step.shape}")

    h_delay = h_step[0, :, :]

    h_delay_clean = h_delay - np.mean(h_delay, axis=0, keepdims=True)

    window = np.hanning(h_delay_clean.shape[0])[:, None]
    h_delay_clean *= window

    h_delay_doppler = np.fft.fft(h_delay_clean, axis=0, norm="ortho")
    h_delay_doppler_shifted = np.fft.fftshift(h_delay_doppler, axes=0)

    tap_indices = np.arange(-6,101)
    delay_bins = tap_indices * cfg.delay_resolution * 1e9  # Convert to nanoseconds

    x, y = np.meshgrid(delay_bins, doppler_bins)
    z_grid_log = 10 * np.log10(np.abs(h_delay_doppler_shifted) + 1e-12)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z_grid_log, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Delay (ns)')
    ax.set_ylabel('Doppler Shift (Hz)')
    ax.set_ylim(-6000,6000)
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(f'Sensor Channel Response - Step {steps}')
    ax.view_init(elev=45, azim=30)
    plt.show()