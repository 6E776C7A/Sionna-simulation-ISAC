import numpy as np
import matplotlib.pyplot as plt

import ofdm_config as cfg

def _steering_vector(azimuth_rad, elevation_rad, antena_positions):
    """
    Compute the steering vector for a given azimuth and elevation.
    
    Parameters:
    - azimuth_rad: Azimuth angle in radians.
    - elevation_rad: Elevation angle in radians.
    - antena_positions: Array of antenna positions.

    Returns:
    - steering_vector: The computed steering vector.
    """
    direction = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
    ])
    phase = (
        2 * np.pi / cfg.wavelength * (
            antena_positions[:, 0] * direction[0] + antena_positions[:, 1] * direction[1] + antena_positions[:, 2] * direction[2]
        )
    )
    return np.exp(1j * phase) / np.sqrt(antena_positions.shape[0])


def compute_music(
    h_delay_doppler_all,
    doppler_idx,
    delay_idx,
    antena_positions,
    num_sources=1,
    doppler_radius=2,
    delay_radius=1,
    azimuth_grid_deg=None,
    elevation_grid_deg=None,
):
    """
    Estimate azimuth and elevation from complex DD data for all antennas.

    h_delay_doppler_all has shape (antennas, Doppler, delay). The selected
    Doppler-delay neighborhood supplies snapshots for the covariance matrix.
    """
    if h_delay_doppler_all.ndim != 3:
        raise ValueError(
            "Expected h_delay_doppler_all with shape "
            "(num_antennas, num_doppler, num_delay)."
        )

    num_antenas, num_doppler, num_delay = h_delay_doppler_all.shape
    antena_positions = np.asarray(antena_positions, dtype=float)

    if antena_positions.shape != (num_antenas, 3):
        raise ValueError(
            "antena_positions must have shape "
            f"({num_antenas}, 3), got {antena_positions.shape}."
        )
    if not 1 <= num_sources < num_antenas:
        raise ValueError("num_sources must be between 1 and num_antenas - 1.")

    doppler_start = max(0, doppler_idx - doppler_radius)
    doppler_stop = min(num_doppler, doppler_idx + doppler_radius + 1)
    delay_start = max(0, delay_idx - delay_radius)
    delay_stop = min(num_delay, delay_idx + delay_radius + 1)

    observations = h_delay_doppler_all[
        :,
        doppler_start:doppler_stop,
        delay_start:delay_stop,
    ].reshape(num_antenas, -1)

    if observations.shape[1] < num_sources:
        raise ValueError("The selected neighborhood has too few observations.")

    observations = observations - np.mean(observations, axis=1, keepdims=True)
    covariance = observations @ observations.conj().T / observations.shape[1]
    covariance = (covariance + covariance.conj().T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    noise_vectors = eigenvectors[:, :-num_sources]
    noise_projector = noise_vectors @ noise_vectors.conj().T

    if azimuth_grid_deg is None:
        azimuth_grid_deg = np.linspace(-90.0, 90.0, 181)
    if elevation_grid_deg is None:
        elevation_grid_deg = np.linspace(-90.0, 90.0, 181)

    azimuth_grid_deg = np.asarray(azimuth_grid_deg, dtype=float)
    elevation_grid_deg = np.asarray(elevation_grid_deg, dtype=float)
    spectrum = np.empty(
        (elevation_grid_deg.size, azimuth_grid_deg.size),
        dtype=float,
    )

    for elevation_index, elevation_deg in enumerate(elevation_grid_deg):
        elevation_rad = np.deg2rad(elevation_deg)
        for azimuth_index, azimuth_deg in enumerate(azimuth_grid_deg):
            steering = _steering_vector(
                np.deg2rad(azimuth_deg),
                elevation_rad,
                antena_positions,
            )
            denominator = np.real(
                steering.conj() @ noise_projector @ steering
            )
            spectrum[elevation_index, azimuth_index] = 1.0 / max(
                denominator,
                np.finfo(float).eps,
            )

    peak_index = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return {
        "azimuth_deg": float(azimuth_grid_deg[peak_index[1]]),
        "elevation_deg": float(elevation_grid_deg[peak_index[0]]),
        "spectrum": spectrum,
        "azimuth_grid_deg": azimuth_grid_deg,
        "elevation_grid_deg": elevation_grid_deg,
        "eigenvalues": eigenvalues,
        "observations": observations,
    }


def visualize_music_spectrum(music_result):
    """Plot the normalized 2D MUSIC spectrum in dB."""
    spectrum = music_result["spectrum"]
    spectrum_db = 10 * np.log10(
        spectrum / np.max(spectrum) + np.finfo(float).eps
    )

    plt.figure(figsize=(9, 6))
    plt.imshow(
        spectrum_db,
        extent=[
            music_result["azimuth_grid_deg"][0],
            music_result["azimuth_grid_deg"][-1],
            music_result["elevation_grid_deg"][0],
            music_result["elevation_grid_deg"][-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=-40,
        vmax=0,
    )
    plt.colorbar(label="MUSIC spectrum [dB]")
    plt.scatter(
        music_result["azimuth_deg"],
        music_result["elevation_deg"],
        color="red",
        marker="x",
        label="MUSIC estimate",
    )
    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Elevation [deg]")
    plt.title("2D MUSIC spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()
