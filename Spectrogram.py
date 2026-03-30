import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def view_spectrogram(iq, fs, nfft=1024, noverlap=None, cmap='viridis', db_scale=True):
    """
    Plot a centered spectrogram for complex IQ samples.
    """
    if noverlap is None:
        noverlap = nfft // 2

    # 1. Compute spectrogram (return_onesided must be False for complex IQ)
    f, t, Sxx = spectrogram(iq, fs=fs, window='hann', nperseg=nfft,
                            noverlap=noverlap, return_onesided=False, mode='complex')

    # 2. Shift the frequencies so 0 Hz is in the center
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # 3. Compute power
    power = np.abs(Sxx)**2
    if db_scale:
        power = 10 * np.log10(power + 1e-12)

    plt.figure(figsize=(10, 6))
    # Use kHz or MHz depending on your sample rate for readability
    plt.pcolormesh(t, f / 1e3, power, shading='auto', cmap=cmap)
    
    plt.ylabel('Frequency [kHz] (Relative to Center)')
    plt.xlabel('Time [s]')
    plt.title('Complex IQ Spectrogram (Centered)')
    plt.colorbar(label='Power [dB]' if db_scale else 'Power')
    
    # 4. Set limits to show the full complex bandwidth
    plt.ylim(-fs/2/1e3, fs/2/1e3) 
    plt.grid(alpha=0.3)
    plt.show()