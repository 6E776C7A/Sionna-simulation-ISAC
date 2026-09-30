import numpy as np
import matplotlib.pyplot as plt
import ofdm_config as cfg

# Wczytanie historii odpowiedzi impulsowej kanału
taps = np.load("historia_taps_dron.npy", allow_pickle=True)

# 1. Osie fizyczne: Opóźnienie i Kąt
tap_indices = np.arange(-6, 101)
delay_bins_ns = (tap_indices * cfg.delay_resolution) / 1e-9

num_angle_bins = 256
sin_vals = np.linspace(-1.0, 1.0, num_angle_bins)
angle_bins_deg = np.arcsin(sin_vals) * (180.0 / np.pi)

# Okna wygładzające
window_sym = np.hanning(cfg.num_symbols)[None, :, None]
# Okno przestrzenne 2D dla całej apertury UPA 4x4: (wiersze_elewacji, kolumny_azymutu, 1)
window_2d_ant = (np.hanning(cfg.rx_rows)[:, None] * np.hanning(cfg.rx_cols)[None, :])[:, :, None]

for step in range(len(taps)):
    a, tau = taps[step]
    
    # h_step: (16 anten, 512 symboli, 107 odczepów)
    h_step = a[0, :, 0, 0, :, :]
    
    # 2. Usuwanie clutteru stacjonarnego (MTI wzdłuż symboli)
    h_clean = h_step - np.mean(h_step, axis=1, keepdims=True)
    
    # 3. FFT Dopplera po osi symboli
    h_doppler = np.fft.fftshift(
        np.fft.fft(h_clean * window_sym, axis=1, norm="ortho"), 
        axes=1
    )
    
    # 4. Tłumienie strefy zerowego Dopplera (pozostałości budynków wokół DC)
    zero_idx = cfg.num_symbols // 2
    h_doppler_moving = h_doppler.copy()
    h_doppler_moving[:, zero_idx - 7 : zero_idx + 8, :] = 0.0
    
    # Oś Dopplera do wykresu
    doppler_bins = np.arange(-cfg.num_symbols // 2, cfg.num_symbols // 2) * cfg.doppler_resolution
    
    # Pusta macierz na globalną mapę Azymut-Doppler (256 kątów, 512 prędkości)
    global_map_power = np.zeros((num_angle_bins, cfg.num_symbols))
    
    # Iterujemy po 107 odczepach odległości
    for tap_idx in range(h_doppler_moving.shape[2]):
        
        h_slice_doppler = h_doppler_moving[:, :, tap_idx]
        
        # Pomijamy tylko absolutne zera (brak jakiegokolwiek sygnału)
        if np.max(np.abs(h_slice_doppler)) == 0.0:
            continue
            
        # Układamy 16 anten dla tego konkretnego dystansu
        h_upa = h_slice_doppler.reshape(cfg.rx_rows, cfg.rx_cols, -1)
        
        # Przestrzenne 2D-FFT
        h_spatial_raw = np.fft.fftshift(
            np.fft.fft2(np.conj(h_upa) * window_2d_ant, s=(num_angle_bins, num_angle_bins), axes=(0, 1)),
            axes=(0, 1)
        )
        
        # Przekładamy osie (Azymut = axis 0, Elewacja = axis 1)
        h_spatial = np.transpose(h_spatial_raw, (1, 0, 2))
        
        # Rzutujemy na Azymut-Doppler (max po Elewacji)
        current_map_power = np.max(np.abs(h_spatial)**2, axis=1)
        
        # Nakładamy wynik na globalną mapę
        global_map_power = np.maximum(global_map_power, current_map_power)
        
    # Zabezpieczenie przed dzieleniem przez zero
    max_val = np.max(global_map_power)
    if max_val == 0.0:
        max_val = 1.0
        
    map_db = 10 * np.log10(global_map_power / max_val + 1e-12)
    
    # Wykres Globalny Angle-Doppler
    plt.figure(figsize=(10, 6))
    plt.imshow(
        map_db, 
        aspect='auto', 
        extent=[doppler_bins[0], doppler_bins[-1], angle_bins_deg[0], angle_bins_deg[-1]],
        origin='lower', 
        cmap='viridis',
        vmin=-12, vmax=0
    )
    plt.colorbar(label='Znormalizowana moc (dB)')
    plt.xlabel('Częstotliwość Dopplera (Hz)')
    plt.xlim(-6000, 6000)
    plt.ylabel('Kąt azymutu (°)')
    plt.title(f'Global Angle-Doppler Map (All Ranges) - Step {step}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()