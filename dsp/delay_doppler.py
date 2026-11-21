import numpy as np
import ofdm_config as cfg
import matplotlib.pyplot as plt
from dsp.mti_filter import mti_filter

def compute_delay_doppler_map(h_cfr, snapshot_id, **mti_filter_params):
    """
    Compute the delay-Doppler map for specific snapshot in the h_cfr data.

    Parameters:
    - h_cfr: Input channel frequency response data.
    - snapshot_id: The ID of the snapshot to process.
    - mti_filter_params: Dictionary containing parameters for the MTI filter (method, cutoff_hz, order, rs, fs, axis).
    Returns:
    - h_delay_doppler: Complex delay-Doppler maps for all antennas.
    - h_delay_doppler_cfar: 2D equivalent-amplitude map obtained from the
      sum of powers across antennas.
    """

    h = h_cfr[snapshot_id,0,:,0,0,:,:]

    print(f"Processing snapshot {snapshot_id + 1} with shape {h.shape}")

    frequency_window = np.zeros(h.shape[2])
    frequency_window[116:116 + 792] = np.blackman(792)

    h = h * frequency_window[None,None,:] 
    h = np.fft.ifftshift(h, axes=2)

    h_delay = np.fft.ifft(h, axis=2, norm='ortho')

    h_delay_active = h_delay[:, :cfg.num_symbols, :]

    mti_filter_params.setdefault('axis', 1)  
    h_delay_filtered = mti_filter(h_delay_active, **mti_filter_params)

    doppler_window = np.blackman(cfg.num_symbols)[None,:,None]
    h_delay_windowed = h_delay_filtered * doppler_window

    h_delay_doppler = np.fft.fft(h_delay_windowed,n=cfg.doppler_fft_size, axis=1, norm='ortho')
    h_delay_doppler = np.fft.fftshift(h_delay_doppler, axes=1)

    power_map = np.sum(np.abs(h_delay_doppler) ** 2, axis=0)
    h_delay_doppler_cfar = np.sqrt(power_map)

    return h_delay_doppler, h_delay_doppler_cfar

def visualize_3d_delay_doppler_map(h_delay_doppler, max_delay_ns=2000.0, max_doppler_hz=6000.0):
    """
    Visualize the delay-Doppler map for each frame in the h_delay_doppler data.

    Parameters:
    - h_delay_doppler: Input delay-Doppler data.
    - max_delay_ns: Maximum delay in nanoseconds for visualization.
    - max_doppler_hz: Maximum Doppler shift in Hz for visualization.
    Returns:
    - None
    """

    x, y = np.meshgrid(cfg.delay_bins, cfg.doppler_bins)

    if max_delay_ns is None:
        delay_end_idx = len(cfg.delay_bins)
    else:
        delay_end_idx = np.searchsorted(cfg.delay_bins, max_delay_ns)

    if max_doppler_hz is None:
        doppler_start_idx = 0
        doppler_end_idx = len(cfg.doppler_bins)
    else:
        doppler_start_idx = np.searchsorted(cfg.doppler_bins, -max_doppler_hz)
        doppler_end_idx = np.searchsorted(cfg.doppler_bins, max_doppler_hz)

    x_grid = x[doppler_start_idx:doppler_end_idx, :delay_end_idx]
    y_grid = y[doppler_start_idx:doppler_end_idx, :delay_end_idx]
    z_slice = h_delay_doppler[doppler_start_idx:doppler_end_idx, :delay_end_idx]
    z_grid_log = 20 * np.log10(np.abs(z_slice) + 1e-12)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_grid, y_grid, z_grid_log, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Delay (ns)')
    ax.set_ylabel('Doppler Shift (Hz)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(f'Delay-Doppler Map')

    plt.show()
    plt.close(fig)

def visualize_magnitude_map(h_delay_doppler):
    """
    Visualize the delay-Doppler map in the h_delay_doppler data.

    Parameters:
    - h_delay_doppler: Input delay-Doppler data.
    Returns:
    - None
    """

    magnitude = np.abs(h_delay_doppler)
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_db, 
            extent=[cfg.delay_bins[0], cfg.delay_bins[-1], cfg.doppler_bins[0], cfg.doppler_bins[-1]], 
            aspect='auto', cmap='viridis', origin='lower', vmax=0, vmin=-60)
    plt.colorbar(label='Magnitude (dB)')
    plt.title(f'Delay-Doppler Map')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Doppler Shift (Hz)')
    plt.xlim(0,2000)
    plt.ylim(-6000,6000)
    plt.show()
    plt.close()

def visualize_power_map(h_delay_doppler):
    """
    Visualize the power delay-Doppler map in the h_delay_doppler data.

    Parameters:
    - h_delay_doppler: Input delay-Doppler data.
    Returns:
    - None
    """

    power = np.abs(h_delay_doppler)**2

    plt.figure(figsize=(10, 6))
    plt.imshow(power, 
            extent=[cfg.delay_bins[0], cfg.delay_bins[-1], cfg.doppler_bins[0], cfg.doppler_bins[-1]], 
            aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Power')
    plt.title(f'Power Delay-Doppler Map')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Doppler Shift (Hz)')
    plt.xlim(0,2000)
    plt.ylim(-6000,6000)
    plt.show()
    plt.close()