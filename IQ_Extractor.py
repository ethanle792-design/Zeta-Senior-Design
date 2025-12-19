import numpy as np
import matplotlib.pyplot as plt

def load_iq(
    filepath,
    dtype="int16",
    swap_iq=False,
    normalize=True,
    remove_dc=False
):
    """
    Load IQ samples from a raw SDR IQ file.

    Parameters
    ----------
    filepath : str
        Path to IQ file (e.g. ".c64", ".iq", etc.)
    dtype : str
        "int16" or "float32"
    swap_iq : bool
        Swap I and Q channels if needed
    normalize : bool
        Normalize int16 to [-1,1]
    remove_dc : bool
        Subtract mean to remove DC offset / LO leakage

    Returns
    -------
    iq : complex ndarray
        Baseband IQ samples
    """

    dtype = dtype.lower()

    # -------------------------
    # Float32 IQ format
    # -------------------------
    if dtype == "float32":
        raw = np.fromfile(filepath, dtype=np.float32)

        I = raw[0::2]
        Q = raw[1::2]

        if swap_iq:
            I, Q = Q, I

        iq = I + 1j * Q

    # -------------------------
    # Int16 IQ format (RTL-SDR default)
    # -------------------------
    elif dtype == "int16":
        raw = np.fromfile(filepath, dtype=np.int16)

        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)

        if swap_iq:
            I, Q = Q, I

        iq = I + 1j * Q

        if normalize:
            iq /= 32768.0

    else:
        raise ValueError("dtype must be 'int16' or 'float32'")

    # -------------------------
    # Remove DC spike
    # -------------------------
    if remove_dc:
        iq -= np.mean(iq)

    return iq

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def extract_offset_signal(iq_samples, sample_rate, beacon_offset_hz, bw_hz):
    """
    Shifts a specific offset frequency to 0 Hz and filters it.
    
    Parameters:
        beacon_offset_hz: Where the beacon is relative to center (e.g., +500e3)
        bw_hz: The bandwidth to keep (e.g., 125e3)
    """
    
    # 1. Generate the Mixing Signal (Complex Sine Wave)
    # We want to shift by -offset to bring it to 0
    t = np.arange(len(iq_samples))
    mixer = np.exp(-1j * 2 * np.pi * (beacon_offset_hz / sample_rate) * t)
    
    # 2. Mix (Shift the spectrum)
    shifted_iq = iq_samples * mixer
    
    # 3. Design Lowpass Filter
    # Now that signal is at 0, we just need a lowpass for half the BW
    cutoff = bw_hz / 2
    num_taps = 101
    taps = signal.firwin(num_taps, cutoff / (sample_rate/2), window='hamming')
    
    # # 4. Filter
    # filtered_iq = signal.lfilter(taps, 1.0, shifted_iq)
    
    # return filtered_iq
    return shifted_iq

# Usage Example:
# SDR tuned to 915.0 MHz. Beacon is at 915.5 MHz.
# offset = 500e3 (500 kHz)
# clean_signal = extract_offset_signal(raw_iq, 2e6, 500e3, 125e3)

def view_iq_time(iq, fs, num_samples):
    """
    Plot raw IQ samples in time domain.

    Parameters:
        iq : complex ndarray
            IQ samples
        fs : float
            Sample rate in Hz
        num_samples : int
            Number of samples to plot
    """
    # Take only first N samples for visualization
    iq_plot = iq[:num_samples]

    t = np.arange(len(iq_plot)) / fs  # time axis in seconds

    plt.figure(figsize=(10,4))
    plt.plot(t, np.real(iq_plot), label='I')
    plt.plot(t, np.imag(iq_plot), label='Q')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Raw IQ Samples (Time Domain)')
    plt.legend()
    plt.grid(True)
    plt.show()