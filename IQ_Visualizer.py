import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.stats import kurtosis

def extract_capture_metadata(json_path):
    """
    Parses the full JSON structure and preserves keys needed for sync.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # We return the whole 'data' dict or a structured version.
    # Let's keep the structure but ensure 'timing' is accessible.
    return data 

def plot_signal_quality(iq, fs, symbol_len, step=None):
    """
    Plots the Magnitude vs. Quality Score (Q = mean/std) over time.
    
    iq: Complex IQ data
    fs: Sample rate
    symbol_len: Number of samples in one LoRa chirp
    step: How often to calculate Q (default is 1/4 of a symbol for smoothness)
    """
    if step is None:
        step = symbol_len // 4
        
    mag = np.abs(iq)
    time_axis = np.arange(len(mag)) / fs
    
    # Calculate sliding window Quality Score
    qualities = []
    quality_times = []
    
    for i in range(0, len(mag) - symbol_len, step):
        window = mag[i : i + symbol_len]
        mu = np.mean(window)
        sigma = np.std(window)
        
        # Q = Mean / StdDev. 
        # A clean ring (mu=1, sigma=small) results in a HIGH Q.
        # Two rings/smearing (high sigma) results in a LOW Q.
        q_score = mu / (sigma + 1e-6)
        
        qualities.append(q_score)
        quality_times.append(time_axis[i + symbol_len // 2])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Raw Magnitude
    ax1.plot(time_axis, mag, alpha=0.5, label='Magnitude')
    ax1.set_ylabel('Absolute Magnitude')
    ax1.set_title('BANSHEE Signal Analysis: Power vs. Quality')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Quality Score
    ax2.plot(quality_times, qualities, color='orange', lw=2, label='Quality Score (Q)')
    ax2.axhline(y=1.41, color='red', linestyle='--', label='Theoretical Max (Clean Chirp)')
    ax2.set_ylabel('Quality Score (mu/sigma)')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage for your BANSHEE rig:
# plot_signal_quality(iq_bb, fs_bb, detector.num_samples_per_symbol)



def calculate_iq_weight(iq_data):
    """
    Specifically rewards constellations with a 'Bright Center' and 'Faint Ring'.
    File 4 will now score the highest.
    """
    mags = np.abs(iq_data)
    if np.max(mags) == 0: return 0
    
    # 1. Define the 'Center' and the 'Ring' zones
    # We define the 'Center' as the inner 15% of the magnitude range
    center_mask = mags < (0.15 * np.max(mags))
    center_count = np.sum(center_mask)
    ring_count = np.sum(~center_mask)
    
    if ring_count == 0: return 0

    # 2. Calculate Central Density Ratio
    # This rewards having WAY more samples at the origin than on the ring.
    # File 4 has a massive center_count and a tiny ring_count.
    density_ratio = center_count / ring_count
    
    # 3. Cleanliness Factor (Ring Thickness)
    # We want the ring samples to have very low variance (a thin line)
    ring_mags = mags[~center_mask]
    ring_thickness = np.std(ring_mags) / (np.mean(ring_mags) + 1e-9)
    
    # 4. Final Weight
    # We reward high center density and PUNISH a thick/blurry ring
    weight = (density_ratio) / (1.0 + ring_thickness)
    
    # Normalize to keep numbers manageable (Optional)
    return np.log1p(weight)

class IQDataManager:
    def __init__(self, metadata):
        self.meta = metadata
        # Accessing actual rate from your JSON structure
        self.fs = metadata.get('rate')  # was metadata['actual']['rate_sps']
        self.fc = metadata.get('freq')  # was metadata['actual']['freq_hz']

    def load_iq(self, filename, format_type='cf32'):
        """
        Loads IQ data and normalizes it to a float range of -1.0 to 1.0.
        Handles both bladeRF (cs16) and standard (cf32) formats.
        """
        if format_type == 'cs16':
            # bladeRF / HackRF int16 format
            # Layout: I, Q, I, Q ... as signed 16-bit integers
            raw = np.fromfile(filename, dtype=np.int16)
            # Convert to float and normalize by max int16 value
            # This ensures the LoRaDetector sees the same 'amplitude' scale as cf32
            raw = raw.astype(np.float32) / 32768.0
            iq = raw[0::2] + 1j * raw[1::2]
            print(f"[*] Loaded CS16 data: {len(iq)} complex samples.")
            
        else:
            # Standard complex float 32-bit format
            iq = np.fromfile(filename, dtype=np.complex64)
            print(f"[*] Loaded CF32 data: {len(iq)} complex samples.")
            
        return iq

    def sync_sensors(self, compass_csv, num_iq_samples, delay_offset=0.0):
        """
        Maps 'heading' from CSV to IQ sample indices using the 't' column.
        """
        import pandas as pd
        df = pd.read_csv(compass_csv)
        
        # Clean whitespace and identify columns
        df.columns = df.columns.str.strip()
        
        # Mapping for your specific CSV structure
        time_col = 't' 
        heading_col = 'heading'

        if time_col not in df.columns:
            raise KeyError(f"Expected column '{time_col}' not found. Columns: {df.columns.tolist()}")

        # Apply the preset offset
        df['t_sync'] = df[time_col] + delay_offset
        
        # Create a time vector for the loaded IQ data
        iq_times = np.arange(num_iq_samples) / self.fs
        
        # Zero-Order Hold: Find the closest heading for every IQ sample
        indices = np.searchsorted(df['t_sync'], iq_times, side='right') - 1
        indices = np.clip(indices, 0, len(df) - 1)
        
        return df[heading_col].values[indices].astype(np.float32)
    
class IQVisualizer:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        if self.debug_mode:
            plt.ion()  # Turn on interactive mode to allow multiple windows

    def plot_spectrum(self, iq, fs, center_freq, title="Spectrum Analysis"):
        if not self.debug_mode: return
        
        # Use a unique window name so it doesn't overwrite
        plt.figure(title, figsize=(10, 5))
        
        chunk_size = min(len(iq), 1000000)
        start = len(iq) // 2
        data_chunk = iq[start : start + chunk_size]
        
        window = np.blackman(len(data_chunk))
        fft_data = np.fft.fftshift(np.fft.fft(data_chunk * window))
        freq_bins = np.fft.fftshift(np.fft.fftfreq(len(data_chunk), 1/fs))
        freq_axis_mhz = (freq_bins + center_freq) / 1e6
        
        mag_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
        mag_db -= np.max(mag_db)
        
        plt.plot(freq_axis_mhz, mag_db, color='tab:cyan')
        plt.axvline(center_freq/1e6, color='red', linestyle='--', alpha=0.5)
        plt.title(title)
        plt.xlabel("MHz")
        plt.ylabel("dB")
        plt.grid(True, alpha=0.3)
        plt.draw()
        plt.pause(0.1) # Small pause to let the OS draw the window

    def plot_time(self, iq, fs, time_start=0, time_end=1, title="Time Domain"):
        if not self.debug_mode: return
        
        num_samples_start = int(time_start * fs)
        num_samples_end = int(time_end * fs)
        plt.figure(title, figsize=(10, 4))
        iq_p = iq[num_samples_start:num_samples_end]
        t = np.arange(len(iq_p)) / fs

        plt.plot(t * 1e3, iq_p.real, label='I', alpha=0.7)
        plt.plot(t * 1e3, iq_p.imag, label='Q', alpha=0.7)
        plt.title(title)
        plt.xlabel("ms")
        plt.legend()
        plt.draw()
        plt.pause(0.1)

    def plot_detections(self, corr, regions, fs, headings, title="Final Analysis"):
        # We turn OFF interactive mode for the final plot so it stays open
        plt.ioff() 
        plt.figure("BANSHEE Results", figsize=(14, 6))
        
        time_axis = np.arange(len(corr)) / fs
        plt.plot(time_axis, corr, color='tab:blue')

        for start, stop in regions:
            t_mid = start / fs
            h_val = float(headings[start])
            peak = np.max(corr[start:stop])
            plt.axvspan(start/fs, stop/fs, color='orange', alpha=0.3)
            plt.text(t_mid, peak, f"{h_val:.1f}°", rotation=45, fontweight='bold')

        plt.title(title)
        plt.xlabel("Seconds")
        plt.show() # This final call keeps ALL windows open
        
    def plot_constellation(self, iq):
        # Assuming 'iq' is your numpy array of complex numbers
        plt.figure(figsize=(8, 8))
        plt.scatter(iq.real, iq.imag, s=1, alpha=0.5)

        # Crucial formatting for RF work
        plt.axhline(0, color='black', lw=1) # X-axis (Real)
        plt.axvline(0, color='black', lw=1) # Y-axis (Imag)
        plt.axis('equal')                   # Keep the circle a circle
        plt.grid(True, linestyle='--')
        plt.xlabel('In-phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('IQ Constellation')
        plt.show()
        
    def plot_histo(self, iq, weight):
        plt.figure(figsize=(8, 8))
        
        # 1. Use a log scale for bins so the 'faint ring' is visible 
        # even if the 'bright center' is extremely dense.
        plt.hexbin(iq.real, iq.imag, gridsize=100, cmap='magma', bins='log')
        
        plt.axis('equal')
        plt.colorbar(label='Log10(Sample Density)')
        plt.grid(alpha=0.3)
        
        # 2. Fix the Annotation
        # We use xycoords='axes fraction' so (0.05, 0.95) is always top-left
        plt.annotate(f'Quality Weight: {weight:.2f}', 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    fontsize=12, 
                    fontweight='bold',
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5)) # Makes it readable against 'magma'

        plt.title('BANSHEE Density-Based IQ Constellation')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        
        # Ensure the plot actually pops up
        plt.draw() 
        plt.show()
            
        