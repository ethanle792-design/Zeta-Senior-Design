import numpy as np
from scipy.signal import firwin, lfilter

def lowpass_filter_iq(iq, fs, f_pass, f_stop, numtaps=101, window='hamming'):
    """
    Apply a low-pass FIR filter to IQ samples.

    Parameters
    ----------
    iq : complex ndarray
        Time-domain IQ samples
    fs : float
        Sampling rate (Hz)
    f_pass : float
        Passband edge frequency (Hz)
    f_stop : float
        Stopband edge frequency (Hz)
    numtaps : int
        Number of filter taps (larger = sharper transition)
    window : str
        Window type for FIR design

    Returns
    -------
    iq_filtered : complex ndarray
        Filtered IQ samples
    """
    # Normalized frequencies (0..1, 1 = Nyquist)
    nyq = fs / 2
    f_pass_norm = f_pass / nyq
    f_stop_norm = f_stop / nyq

    # Transition width = stopband - passband
    # Design FIR filter
    taps = firwin(numtaps, cutoff=f_pass_norm, window=window, pass_zero=True)

    # Apply filter to both I and Q separately
    i_filtered = lfilter(taps, 1.0, np.real(iq))
    q_filtered = lfilter(taps, 1.0, np.imag(iq))

    iq_filtered = i_filtered + 1j * q_filtered

    return iq_filtered
