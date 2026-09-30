import numpy as np
import matplotlib.pyplot as plt
import ofdm_config as cfg

# Wczytanie historii odpowiedzi impulsowej kanału ze scenariusza
taps = np.load("historia_taps.npy", allow_pickle=True)

# 1. Definicja fizycznych osi odległości i kąta azymutu
tap_indices = np.arange(-6, 101)
delay_bins_ns = (tap_indices * cfg.delay_resolution) / 1e-9

num_angle_bins = 256
sin_vals = np.linspace(-1.0, 1.0, num_angle_bins)
angle_bins_deg = np.arcsin(sin_vals) * (180.0 / np.pi)

# Okna wygładzające
window_sym = np.hanning(cfg.num_symbols)[None, :, None]
window_ant = np.hanning(4)[:, None]

for step in range(len(taps)):
    a, tau = taps[step]
    
    # Kształt a: (1, 16, 1, 1, 512, 107) -> wybieramy 16 anten
    h_step = a[0, :, 0, 0, :, :]
    
    # 2. Usuwanie clutteru stacjonarnego (MTI)
    h_clean = h_step - np.mean(h_step, axis=1, keepdims=True)
    
    # 3. FFT Dopplera po osi symboli
    h_doppler = np.fft.fftshift(
        np.fft.fft(h_clean * window_sym, axis=1, norm="ortho"), 
        axes=1
    )
    
    # 4. Tłumienie strefy zerowego Dopplera (pozostałości po budynkach)
    zero_idx = cfg.num_symbols // 2
    h_doppler_moving = h_doppler.copy()
    h_doppler_moving[:, zero_idx - 7 : zero_idx + 8, :] = 0.0
    
    # 5. Detekcja składowej ruchomej
    power_dop = np.sum(np.abs(h_doppler_moving)**2, axis=(0, 2))
    best_dop = np.argmax(power_dop)
    
    # 6. Wycięcie plastra dla prędkości celu na 4 antenach azymutalnych (ULA)
    h_az_line = h_doppler_moving[:4, best_dop, :]
    
    # 7. FFT przestrzenne wzdłuż azymutu
    h_azimuth = np.fft.fftshift(
        np.fft.fft(h_az_line * window_ant, n=num_angle_bins, axis=0), 
        axes=0
    )
    
    # 8. Obliczenie macierzy mocy Range-Azimuth
    map_power = np.abs(h_azimuth)**2
    map_db = 10 * np.log10(map_power / np.max(map_power) + 1e-12)
    
    # 9. Precyzyjny odczyt kąta dla echa bezpośredniego (~240 ns) i wielodrogowości (~594 ns)
    direct_tap_idx = np.argmin(np.abs(delay_bins_ns - 240))
    angle_direct = angle_bins_deg[np.argmax(map_power[:, direct_tap_idx])]
    
    multipath_tap_idx = np.argmin(np.abs(delay_bins_ns - 594))
    angle_multipath = angle_bins_deg[np.argmax(map_power[:, multipath_tap_idx])]
    
    print(f"\n--- KROK {step} (Doppler bin: {best_dop}) ---")
    print(f"Echo bezpośrednie celu (~240 ns): Kąt azymutu = {angle_direct:.2f}°")
    print(f"Echo wielodrogowe      (~594 ns): Kąt azymutu = {angle_multipath:.2f}°")
    
    # 10. Wykres Range-Azimuth z zawężoną dynamiką (ujawnia zwarty kształt piku)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        map_db, 
        aspect='auto', 
        extent=[delay_bins_ns[0], delay_bins_ns[-1], angle_bins_deg[0], angle_bins_deg[-1]],
        origin='lower', 
        cmap='viridis',
        vmin=-12, vmax=0
    )
    plt.colorbar(label='Znormalizowana moc (dB)')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Kąt azymutu (°)')
    plt.title(f'Range-Azimuth Map - Step {step} (Doppler bin: {best_dop})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()