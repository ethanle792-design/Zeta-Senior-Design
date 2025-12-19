import matplotlib.pyplot as plt
from IQ_Extractor import load_iq
from IQ_Extractor import extract_offset_signal
from IQ_Extractor import view_iq_time
from Lowpass_Filter import lowpass_filter_iq
from Spectrogram import view_spectrogram
from Bandpass_Filter import apply_fir_filter
from Matched_Filter import matched_filter
from Matched_Filter import extract_corr_regions
from Matched_Filter import generate_lora_chirp
import numpy as np

# -------------------------
# Hard-code your IQ file here
# -------------------------
IQ_FILE = "/Users/ethanle/Downloads/915_3Msps.c64"
FS = 3e6   # sample rate

# Load IQ
iq = load_iq(
    filepath=IQ_FILE,
    dtype="float32",   # or "float32"
    swap_iq=False,
    normalize=True,
    remove_dc=True
)   

print("Loaded samples:", len(iq))
num_samples = len(iq)

iq_shifted = extract_offset_signal(
            iq_samples=iq,
            sample_rate=FS,
            beacon_offset_hz=200e3,
            bw_hz=125e3
)

# -------------------------
# Plot DFT
# -------------------------
# define your hardware tuning frequency
center_freq = 915e6  # Example: 915 MHz

fft_original = np.fft.fftshift(np.fft.fft(iq))
fft_shifted  = np.fft.fftshift(np.fft.fft(iq_shifted))

freq_bins = np.fft.fftshift(np.fft.fftfreq(len(iq), 1/FS))
freq_axis_mhz = (freq_bins + center_freq) / 1e6

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # --- Plot 1: Original IQ ---
# ax1.plot(freq_axis_mhz, 20*np.log10(np.abs(fft_original) + 1e-12), color='tab:red')
# ax1.set_title("Original IQ Spectrum (Before Shift)")
# ax1.set_ylabel("Magnitude (dB)")
# ax1.grid(True, alpha=0.3)
# # Add a marker for the center frequency
# ax1.axvline(center_freq/1e6, color='black', linestyle='--', label='Center Freq (LO)')
# ax1.legend()

# # --- Plot 2: Shifted IQ ---
# ax2.plot(freq_axis_mhz, 20*np.log10(np.abs(fft_shifted) + 1e-12), color='tab:blue')
# ax2.set_title("Shifted IQ Spectrum (After Mixing)")
# ax2.set_xlabel("Frequency (MHz)")
# ax2.set_ylabel("Magnitude (dB)")
# ax2.grid(True, alpha=0.3)
# ax2.axvline(center_freq/1e6, color='black', linestyle='--', label='Center Freq (LO)')
# ax2.legend()

# plt.tight_layout()
# plt.show()

# view_iq_time(iq, fs=FS, num_samples=num_samples)

# Example parameters
f_pass = 500e3   # 100 kHz passband
f_stop = 600e3   # 120 kHz stopband
numtaps = 201    # sharper filter

iq_filt = lowpass_filter_iq(iq_shifted, FS, f_pass, f_stop, numtaps=numtaps)

# Optional: plot spectrum of filtered signal
fft = np.fft.fftshift(np.fft.fft(iq_filt))
freq = np.fft.fftshift(np.fft.fftfreq(len(iq_filt), 1/FS))

import matplotlib.pyplot as plt
plt.plot(freq/1e3, 20*np.log10(np.abs(fft)+1e-12))
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Filtered IQ Spectrum")
plt.grid()
plt.show()

# Assuming iq is loaded and FS is sample rate
# view_spectrogram(iq, fs=FS, nfft=2048, noverlap=1024)
# view_spectrogram(iq_filt, fs=FS, nfft=2048, noverlap=1024)

# --- CONFIGURATION ---
SF = 10              # Your Beacon Spreading Factor
BW = 125e3           # Your Beacon Bandwidth
FS = 2e6             # Your SDR Sample Rate (Must match your capture!)

# A. Generate Template (Do this ONCE, outside your rotation loop)
chirp_template = generate_lora_chirp(SF, BW, FS)

# --- START ROTATION LOOP HERE ---
# assuming 'blue_iq_data' is your filtered/shifted signal from the previous step

# B. Run Filter
correlation_result = matched_filter(chirp_template, iq_filt)

# C. Extract Magnitude
correlation_mag = np.abs(correlation_result)

import matplotlib.pyplot as plt
plt.plot(correlation_mag)
plt.title(f"Correlation at Angle")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Strength (Raw)")
plt.grid(True)
plt.show()

extract_corr_regions(correlation_mag, threshold=0.9)

# # D. Get Signal Strength Score
# # We take the maximum peak found in this capture window
# peak_strength = np.max(correlation_mag)

# # E. (Optional) Noise Floor Check
# # Find the average "background" level to calculate SNR
# avg_noise = np.mean(correlation_mag)
# snr_score = peak_strength / avg_noise

# print(peak_strength)
# print()

# print(f"Angle: {current_angle_deg} | Raw Strength: {peak_strength:.2f} | SNR: {snr_score:.2f}")

# Store this 'peak_strength' for your Line of Bearing map
# results.append((current_angle_deg, peak_strength))