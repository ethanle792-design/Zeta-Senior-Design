import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

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

class IQDataManager:
    def __init__(self, metadata):
        self.meta = metadata
        # Accessing actual rate from your JSON structure
        self.fs = metadata['actual']['rate_sps']
        self.start_time = metadata['timing']['t0_monotonic']

    def load_iq(self, file_path, format_type='cf32'):
        if format_type == 'cf32':
            return np.fromfile(file_path, dtype=np.complex64)
        elif format_type == 'cs16':
            raw = np.fromfile(file_path, dtype=np.int16)
            float_data = raw.astype(np.float32) / 32768.0
            return float_data[0::2] + 1j * float_data[1::2]
        else:
            raise ValueError("Format must be 'cf32' or 'cs16'")

    def sync_sensors(self, compass_csv, num_iq_samples, delay_offset=0.0):
        df = pd.read_csv(compass_csv)
        df['t_sync'] = df['t_rel_s'] + delay_offset
        
        iq_times = np.arange(num_iq_samples) / self.fs
        
        # Use 'side=left' and clip to ensure we stay within valid CSV indices
        indices = np.searchsorted(df['t_sync'], iq_times, side='left') - 1
        indices = np.clip(indices, 0, len(df) - 1)
        
        # Explicitly return as a numpy array of floats
        return df['heading_deg'].values[indices].astype(np.float32)
    
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

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

    def plot_time(self, iq, fs, num_samples=50000, title="Time Domain"):
        if not self.debug_mode: return
        
        plt.figure(title, figsize=(10, 4))
        iq_p = iq[:num_samples]
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