import numpy as np

# Parametry środowiska
fc = 28e9  # Częstotliwość nośna (Hz)
c = 3e8  # Prędkość światła (m/s)
num_subcarriers = 1024  # Liczba podnośnych
num_symbols = 512   # Liczba symboli OFDM
subcarrier_spacing = 120e3  # Odstęp między podnośnymi (Hz)
cyclic_prefix_length = 72  # Długość prefiksu cyklicznego (próbki)
rx_cols = 4  # Liczba odbiorników w osi X
rx_rows = 4  # Liczba odbiorników w osi Y
num_rx_antennas = rx_cols * rx_rows  # Całkowita liczba anten odbiorczych

# Obliczenia związane z OFDM
wavelength = c / fc  # Długość fali (m)
bandwidth = num_subcarriers * subcarrier_spacing  # Szerokość pasma (Hz)
sample_duration = 1 / bandwidth  # Czas trwania próbki (s)

antena_spacing = wavelength / 2  # Odstęp między antenami (m)

angle_resolution_rad = wavelength / (rx_cols * antena_spacing)  # Rozdzielczość kątowa w radianach
angle_resolution_deg = np.degrees(angle_resolution_rad)  # Rozdzielczość kątowa w stopniach

useful_symbol_duration = 1 / subcarrier_spacing  # Czas trwania użytecznego symbolu (s)
cp_duration = cyclic_prefix_length * sample_duration  # Czas trwania prefiksu cyklicznego (s)
total_symbol_duration = useful_symbol_duration + cp_duration  # Całkowity czas trwania symbolu (s)
ofdm_symbol_rate = 1 / total_symbol_duration  # Szybkość symboli OFDM (symbol/s)

range_resolution = c / (2 * bandwidth)  # Rozdzielczość odległości (m)
delay_resolution = sample_duration  # Rozdzielczość opóźnienia (s)

frame_duration = num_symbols * total_symbol_duration  # Czas trwania ramki (s)
doppler_resolution = 1 / frame_duration  # Rozdzielczość Dopplera (Hz)
velocity_resolution = (doppler_resolution * wavelength) / 2  # Rozdzielczość prędkości (m/s)
max_velocity = (ofdm_symbol_rate * wavelength) / 4  # Maksymalna prędkość (m/s)