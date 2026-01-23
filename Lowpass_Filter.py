import numpy as np
from scipy.signal import firwin, filtfilt

def lowpass_filter_iq(iq, fs, f_pass, numtaps=101, window='hamming'):
    nyq = fs / 2
    f_pass_norm = f_pass / nyq
    
    # Design taps
    taps = firwin(numtaps, cutoff=f_pass_norm, window=window)

    # filtfilt handles complex iq directly and ensures zero time-delay
    # This is critical for accurate geolocation timing!
    return filtfilt(taps, 1.0, iq)

def ddc_shift(iq_samples, fs, offset_hz):
    """
    Shifts the target frequency to 0 Hz using complex mixing.
    
    Parameters:
        iq_samples: Raw complex64 array
        fs: Sampling rate (e.g., 3e6)
        offset_hz: Target Freq - Center Freq (e.g., -400e3)
    """
    # Create time vector normalized to fs for floating point precision
    t = np.arange(len(iq_samples)) / fs
    
    # Generate the mixer (complex exponential)
    # Multiplying by -offset_hz "un-spins" the target back to 0 Hz
    mixer = np.exp(-1j * 2 * np.pi * offset_hz * t)
    
    return iq_samples * mixer

def decimate(iq, fs, M=10):
    # Keep every M-th sample
    decimated_iq = iq[::M]
    new_fs = fs / M
    # Debug print to console so you see it working
    # print(f"--- Decimation Report ---")
    # print(f"Input Samples:  {len(iq)}")
    # print(f"Output Samples: {len(decimated_iq)} (Factor of {M})")
    # print(f"New Sample Rate: {new_fs/1e3} kHz")
    
    return decimated_iq, new_fs