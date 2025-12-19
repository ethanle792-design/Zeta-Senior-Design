import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def view_spectrogram(iq, fs, nfft=1024, noverlap=None, cmap='viridis', db_scale=True):
    """
    Plot a spectrogram of IQ samples using scipy.signal.spectrogram.
    """
    if noverlap is None:
        noverlap = nfft // 2

    # Compute spectrogram (complex IQ)
    f, t, Sxx = spectrogram(iq, fs=fs, window='hann', nperseg=nfft,
                            noverlap=noverlap, mode='complex')

    # Compute power
    power = np.abs(Sxx)**2

    if db_scale:
        power = 10 * np.log10(power + 1e-12)

    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f/1e3, power, shading='auto', cmap=cmap)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.title('IQ Spectrogram')
    plt.colorbar(label='Power [dB]' if db_scale else 'Power')
    plt.ylim(0, fs/2/1e3)
    plt.show()
