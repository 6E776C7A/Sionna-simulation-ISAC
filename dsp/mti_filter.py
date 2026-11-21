from scipy import signal
import numpy as np

def mti_filter(data, method, cutoff_hz=None, order=None, rs=None, fs=None, axis=1):
    """
    Apply a Moving Target Indicator (MTI) filter to the input data.

    Parameters:
    - data: Input data array to be filtered. 
    - method: Filtering method ('cheby2', 'butter', 'dc_recursive' or 'dc_mean').
    - axis: Axis along which to apply the filter.
    - cutoff_hz: Cutoff frequency in Hz for the high-pass filter.
    - order: Order of the filter.
    - rs: Stopband attenuation in dB for Chebyshev Type II filter.

    Returns:
    - Filtered data array.
    """

    if method == 'cheby2':

        if cutoff_hz is None or order is None or rs is None or fs is None:
            raise ValueError("For 'cheby2' method, cutoff_hz, order, rs, and fs must be specified.")
        if fs <= 0:
            raise ValueError("fs must be a positive value representing the sampling frequency.")
        if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
            raise ValueError("cutoff_hz must be between 0 and Nyquist frequency (fs/2).")
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive integer.")
        if rs <= 0:
            raise ValueError("rs must be a positive value for stopband attenuation.")
        
        # Normalize the cutoff frequency
        normalized_cutoff = cutoff_hz / (0.5 * fs)
        # Design a Chebyshev Type II high-pass filter
        sos = signal.cheby2(order, rs, normalized_cutoff, btype='highpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=axis)

    elif method == 'butter':

        if cutoff_hz is None or order is None or fs is None:
            raise ValueError("For 'butter' method, cutoff_hz, order, and fs must be specified.")
        if fs <= 0:
            raise ValueError("fs must be a positive value representing the sampling frequency.")
        if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
            raise ValueError("cutoff_hz must be between 0 and Nyquist frequency (fs/2).")
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive integer.")
        
        # Normalize the cutoff frequency
        normalized_cutoff = cutoff_hz / (0.5 * fs)
        # Design a Butterworth high-pass filter
        sos = signal.butter(order, normalized_cutoff, btype='highpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=axis)

    elif method == 'dc_recursive':

        if cutoff_hz is None or fs is None:
            raise ValueError("For 'dc_recursive' method, cutoff_hz and fs must be specified.")
        if fs <= 0:
            raise ValueError("fs must be a positive value representing the sampling frequency.")
        if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
            raise ValueError("cutoff_hz must be between 0 and Nyquist frequency (fs/2).")
        
        # DC blocking filter: recursive implementation
        alpha = np.exp(-2 * np.pi * cutoff_hz / fs)
        b = [1, -1]
        a = [1, -alpha]
        filtered_data = signal.filtfilt(b, a, data, axis=axis)

    elif method == 'dc_mean':

        # DC blocking filter: subtract the mean along the specified axis
        data_mean = np.mean(data, axis=axis, keepdims=True)
        filtered_data = data - data_mean

    else:
        
        raise ValueError("Invalid method specified. Use 'cheby2', 'butter', 'dc_recursive' or 'dc_mean'.")

    return filtered_data

import ofdm_config as cfg

def get_mti_filter_params(chosen_filter_method, cutoff_hz=300.0):
    """
    Returns the MTI filter parameters based on the chosen filter method.

    Parameters:
    - chosen_filter_method: str, the chosen filter method ('dc_mean', 'dc_recursive', 'butter', 'cheby2')
    - cutoff_hz: float, cutoff frequency for the filter

    Returns:
    - mti_filter_params: dict, containing the parameters for the chosen filter method
    """

    fs = 1 / cfg.total_symbol_duration  # Sampling frequency based on total symbol duration

    match chosen_filter_method:
        case 'dc_mean':
            mti_filter_params = {
                'method': 'dc_mean'
            }
        case 'dc_recursive':
            mti_filter_params = {
                'method': 'dc_recursive',
                'fs': fs,
                'cutoff_hz': cutoff_hz
            }
        case 'butter':
            mti_filter_params = {
                'method': 'butter',
                'fs': fs,
                'cutoff_hz': cutoff_hz,
                'order': 4
            }
        case 'cheby2':
            mti_filter_params = {
                'method': 'cheby2',
                'fs': fs,
                'cutoff_hz': cutoff_hz,
                'order': 4,
                'rs': 60
            }
        case _:
            raise ValueError(f"Unknown filter method: {chosen_filter_method}")

    return mti_filter_params