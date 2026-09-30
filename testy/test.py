import numpy as np
import matplotlib.pyplot as plt
import ofdm_config as cfg

taps = np.load("historia_taps.npy", allow_pickle=True)

# Sprawdzamy Step 4
a, tau = taps[4]
h_step = a[0, :, 0, 0, :, :]  # Kształt: (16 anten, 512 symboli, 107 odczepów)

# 1. Usuwamy clutter
h_clean = h_step - np.mean(h_step, axis=1, keepdims=True)

# 2. Doppler FFT
window_sym = np.hanning(cfg.num_symbols)[None, :, None]
h_doppler = np.fft.fftshift(np.fft.fft(h_clean * window_sym, axis=1, norm="ortho"), axes=1)

# Znajdujemy indeks odczepu opóźnienia i prędkości z maksymalną energią
power_map = np.sum(np.abs(h_doppler)**2, axis=0)  # Kształt: (512, 107)
best_dop, best_tap = np.unravel_index(np.argmax(power_map), power_map.shape)

print(f"Pik wykryty w: Doppler bin = {best_dop}, Tap index = {best_tap}")
print(f"Moc piku przed normalizacją: {power_map[best_dop, best_tap]}")

# Wyciągamy wektor 16 anten dokładnie z tego jednego piku
ant_vector = h_doppler[:, best_dop, best_tap]  # Kształt: (16,)

# Wyświetlamy fazy na 16 antenach
plt.figure(figsize=(8, 4))
plt.plot(np.angle(ant_vector), 'o-')
plt.title("Faza na kolejnych 16 antenach dla piku odbicia")
plt.xlabel("Indeks anteny (0-15)")
plt.ylabel("Faza (rad)")
plt.grid(True)
plt.show()

# ant_vector ma rozmiar (16,) z poprzedniego testu
# Przearanżowujemy na 4 wiersze i 4 kolumny
ant_matrix = ant_vector.reshape(4, 4)

# Robimy FFT po kolumnach (Azymut) z paddingiem
num_angle_bins = 256
window = np.hanning(4)[:, None]

# FFT 2D przestrzenne: wiersze (elewacja), kolumny (azymut)
spatial_spectrum = np.fft.fftshift(np.fft.fft2(ant_matrix, s=(num_angle_bins, num_angle_bins)))
spatial_power = np.abs(spatial_spectrum)**2
spatial_db = 10 * np.log10(spatial_power / np.max(spatial_power))

# Oś kątowa dla d = lambda/2
sin_vals = np.linspace(-1, 1, num_angle_bins)
angles_deg = np.arcsin(sin_vals) * (180.0 / np.pi)

# Profil azymutalny (rzutowanie na azymut - bierzemy max po elewacji)
azimuth_profile = np.max(spatial_db, axis=0)

plt.figure(figsize=(9, 4))
plt.plot(angles_deg, azimuth_profile, lw=2)
plt.title(f"Profil kątowy azymutu dla piku celu (Krok 4)")
plt.xlabel("Kąt azymutu (°)")
plt.ylabel("Znormalizowana moc (dB)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([-30, 2])
plt.show()

peak_angle = angles_deg[np.argmax(azimuth_profile)]
print(f"Szacowany kąt azymutu: {peak_angle:.2f}°")

# Weź tylko pierwsze 4 anteny (pojedynczy rząd ULA w azymucie)
ula_vector = ant_vector[:4]

num_angle_bins = 256
window = np.hanning(4)

# 1D FFT wzdłuż azymutu
spec = np.fft.fftshift(np.fft.fft(ula_vector * window, n=num_angle_bins))
spec_db = 10 * np.log10(np.abs(spec)**2 / np.max(np.abs(spec)**2))

sin_vals = np.linspace(-1, 1, num_angle_bins)
angles_deg = np.arcsin(sin_vals) * (180.0 / np.pi)

plt.figure(figsize=(9, 4))
plt.plot(angles_deg, spec_db, lw=2)
plt.title("Profil azymutu z podmacierzy 1x4")
plt.xlabel("Kąt azymutu (°)")
plt.ylabel("Znormalizowana moc (dB)")
plt.grid(True)
plt.ylim([-30, 2])
plt.show()

print(f"Pik azymutu: {angles_deg[np.argmax(spec_db)]:.2f}°")