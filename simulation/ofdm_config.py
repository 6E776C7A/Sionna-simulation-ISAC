import numpy as np

# Parametry środowiska i sygnału 
fc = 28e9  # Częstotliwość nośna 
c = 3e8  # Prędkość światła
num_subcarriers = 792  # Liczba aktywnych podnośnych
fft_size = 1024  # Rozmiar FFT
num_symbols = 252   # Liczba symboli OFDM w ramce 
doppler_fft_size = 256  # Rozmiar FFT Dopplera
subcarrier_spacing = 120e3  # Odstęp między podnośnymi
cyclic_prefix_length = 72  # Długość CP

# Geometria szyku antenowego
rx_cols = 4  # Liczba anten w osi X (azymut)
rx_rows = 4  # Liczba anten w osi Y (elewacja)
num_rx_antennas = rx_cols * rx_rows

# 1. Fizyka fali i próbkowanie
wavelength = c / fc  # Długość fali 
sampling_rate = fft_size * subcarrier_spacing  # Częstotliwość próbkowania 
sample_duration = 1 / sampling_rate  # Czas trwania jednej próbki 
bandwidth = num_subcarriers * subcarrier_spacing  # Efektywne pasmo sygnału 

# 2. Anteny i rozdzielczość kątowa
antena_spacing = wavelength / 2  # Optymalny odstęp anten pół fali 
angle_resolution_rad = wavelength / (rx_cols * antena_spacing)  # Rozdzielczość w radianach
angle_resolution_deg = np.degrees(angle_resolution_rad)  # Rodzielczość w stopniach

# 3. Parametry czasowe OFDM
useful_symbol_duration = 1 / subcarrier_spacing  # T_fft 
cp_duration = cyclic_prefix_length * sample_duration  # T_cp 
total_symbol_duration = useful_symbol_duration + cp_duration  # T_sym 
ofdm_symbol_rate = 1 / total_symbol_duration  # Szybkość symboli 

# 4. Rozdzielczości radarowe
range_resolution = c / (2 * bandwidth)  # Rozdzielczość odległości 
delay_resolution = sample_duration  # Rozdzielczość opóźnienia

frame_duration = num_symbols * total_symbol_duration  # Czas trwania ramki danych 
doppler_resolution = 1 / frame_duration  # Rozdzielczość Dopplera 
velocity_resolution = (doppler_resolution * wavelength) / 2  # Rozdzielczość prędkości 
max_velocity = wavelength / (4 * total_symbol_duration)  # Maksymalna jednoznaczna prędkość 

# 5. Okno przetwarzania (Range FFT)
R_max = 320.0  # Maksymalny zakładany zasięg radaru (m)
# Maksymalne opóźnienie w próbkach (2 * R_max / c) / sample_duration
l_max_dynamic = int(np.ceil((2 * R_max * sampling_rate) / c))  # Wielkość okna czasowego w próbkach

doppler_bins = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, d=total_symbol_duration)) # Przestrzeń Dopplera
delay_bins = np.arange(fft_size) * delay_resolution * 1e9 # Opoźnienia w nanosekundach