import matplotlib.pyplot as plt
from Lowpass_Filter import lowpass_filter_iq
from Matched_Filter import matched_filter
from Matched_Filter import extract_corr_regions
from Matched_Filter import generate_lora_chirp
from Lowpass_Filter import ddc_shift
from IQ_Extractor import plot_debug_spectrum
from Lowpass_Filter import decimate
import numpy as np

# --- CONFIGURATION ---
SF = 12              # Your Beacon Spreading Factor
BW = 125e3           # Your Beacon Bandwidth
FS = 3e6   # sample rate
center_freq = 915e6  # SDR tuning
lora_freq = 915.5e6  # Centered at Lora chrip
offset_hz = lora_freq - center_freq

# Load IQ
iq = np.fromfile("fw_915_stop_turns.c64", dtype=np.complex64, count=40000000)

print("Loaded samples:", len(iq))

iq_shifted = ddc_shift(iq, FS, offset_hz)

# Debug
# plot_debug_spectrum(iq, FS, center_freq, title="Spectrum")
# plot_debug_spectrum(iq_shifted, FS, center_freq, title="shift Spectrum")

# Filter parameters
f_pass = 80e3   # passband
numtaps = 101    # sharper filter

iq_filt = lowpass_filter_iq(iq_shifted, FS, f_pass, numtaps=numtaps)
downsampled_iq_filt, downsampled_fs = decimate(iq_filt, FS)


# Optional: plot spectrum of filtered signal
# plot_debug_spectrum(iq_filt, FS, lora_freq, title="Filtered spectrum", chunk_size=100e3)
# plot_debug_spectrum(downsampled_iq_filt, downsampled_fs, lora_freq, title="downsampled Filtered spectrum", chunk_size=100e3)

# A. Generate Template
chirp_template = generate_lora_chirp(SF, BW, downsampled_fs)

# B. Run Filter
correlation_result = matched_filter(chirp_template, downsampled_iq_filt)

# C. Extract Magnitude
correlation_mag = np.abs(correlation_result)

time_axis = np.arange(len(correlation_mag)) / downsampled_fs

# Plotting
plt.plot(time_axis, correlation_mag)
plt.title(f"Correlation at Angle")
plt.xlabel("Time (s)")
plt.ylabel("Correlation Strength (Raw)")
plt.grid(True)
plt.show()


# 1. Run your detection logic first
threshold = 0.7
regions = extract_corr_regions(correlation_mag, threshold=threshold)

# 1. Setup the Plot
plt.figure(figsize=(15, 6))
plt.plot(time_axis, correlation_mag, label='Correlation Magnitude', color='tab:blue', alpha=0.8)

# 2. Overlay the Regions
for i, (start, stop) in enumerate(regions):
    # --- THE CRITICAL FIX: Convert indices to time ---
    t_start = start / downsampled_fs
    t_stop  = stop / downsampled_fs
    t_mid   = (t_start + t_stop) / 2
    
    # Draw the span using TIME coordinates
    plt.axvspan(t_start, t_stop, color='orange', alpha=0.3)
    
    # Draw the text using TIME coordinates
    peak_val = np.max(correlation_mag[start:stop])
    plt.text(t_mid, peak_val, f"Peak: {peak_val:.2f}", 
             ha='center', va='bottom', fontsize=8)

# 3. Force the X-axis to stay within the time range
plt.xlim(time_axis[0], time_axis[-1])

plt.title(f"BANSHEE Detection Overlay | {len(regions)} Chirps Found")
plt.xlabel("Time (s)")
plt.ylabel("Correlation Strength")
plt.grid(True, alpha=0.3)
plt.show()