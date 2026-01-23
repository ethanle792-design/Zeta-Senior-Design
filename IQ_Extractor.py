import numpy as np
import matplotlib.pyplot as plt

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
    


def plot_debug_spectrum(iq, fs, center_freq, title="Spectrum Analysis", chunk_size=1e6  ):
    """
    Plots the PSD of a chunk of IQ data with a centered RF axis.
    
    Parameters:
        iq: Complex IQ samples
        fs: Sample rate (Hz)
        center_freq: The RF frequency that '0 Hz' represents (Hz)
        title: Title for the plot
        chunk_size: Number of samples to FFT (prevents memory crash)
    """
    # 1. Take a manageable chunk from the middle of the data
    # (Middle is usually more stable than the very beginning of a recording)
    # Ensure start and end are forced to integers
    start = int(len(iq) // 2)
    end = int(start + chunk_size)
    
    # Now this slice will work every time
    data_chunk = iq[start:end]
    
    # 2. Perform FFT
    # Use a Window to prevent spectral leakage (makes the CW spike sharper)
    window = np.blackman(len(data_chunk))
    fft_data = np.fft.fftshift(np.fft.fft(data_chunk * window))
    
    # 3. Create Frequency Axis
    freq_bins = np.fft.fftshift(np.fft.fftfreq(len(data_chunk), 1/fs))
    freq_axis_mhz = (freq_bins + center_freq) / 1e6
    
    # 4. Calculate Magnitude in dB
    # We normalize to the peak (0 dB) so it's easy to see the SNR
    mag_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
    mag_db -= np.max(mag_db)
    
    # 5. Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(freq_axis_mhz, mag_db, color='tab:blue', linewidth=1)
    
    # Add Marker for the "Logical Center"
    plt.axvline(center_freq/1e6, color='red', linestyle='--', alpha=0.5, label='Center/LO')
    
    plt.title(title)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Magnitude (dB)")
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim(-80, 5) # Show 80dB of dynamic range
    plt.legend()
    plt.show()

    # Find and print the peak frequency for the console
    peak_idx = np.argmax(mag_db)
    print(f"[{title}] Peak found at: {freq_axis_mhz[peak_idx]:.6f} MHz")