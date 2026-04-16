import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager, plot_signal_quality
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram


# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "metadata_file": "FlightTests/spin_capture_0.json",
    "iq_format": "cs16",
    "target_freq": 915.5e6,
    "lora_sf": 12,
    "lora_bw": 125e3,
    "decimation": 4,
    "threshold": 0.34,
    "window_size": 5,
    "debug_plots": False
}

def main():
    plt.close('all')



    viz = IQVisualizer(debug_mode=CONFIG["debug_plots"])



    fs_actual = 915e6
    fc_actual = 915.5e6

    filename = 'Apr6 Fields/Spot0.cs16'

    for i in range(1):
        # 1. Read the file as 16-bit signed integers (int16)
        # .cs16 is interleaved: I, Q, I, Q, ...
        raw = np.fromfile(filename, dtype=np.int16)
            # Convert to float and normalize by max int16 value
            # This ensures the LoRaDetector sees the same 'amplitude' scale as cf32
        raw = raw.astype(np.float32) / 32768.0
        iq = raw[0::2] + 1j * raw[1::2]
        print(f"[*] Loaded CS16 data: {len(iq)} complex samples.")


        # iq = iq[i * int(len(iq)/3) : (i+1) * int(len(iq)/3)]
        # iq = (iq - np.mean(iq)) / np.std(iq)
        # # viz.plot_constellation(iq)
        # viz.plot_histo(iq, 1)
        # viz.plot_time(iq, fs_actual, 7.9, 8.1)

        # viz.plot_spectrum(iq, fs_actual, fc_actual, "1. Raw Capture Spectrum")
        # view_spectrogram(iq, fs_actual)


    # 4. DSP Pipeline


    print(f"[*] Running Down-Conversion Pipeline...")
    sdc = SignalDownConverter(fs=fs_actual)

    # Calculate offset: (Requested Center - Actual Tuned Center)
    freq_offset = CONFIG["target_freq"] - fc_actual

    iq_bb, fs_bb = sdc.process_pipeline(
        iq,
        offset_hz=freq_offset,
        bw_target=CONFIG["lora_bw"],
        decimation_factor=CONFIG["decimation"]
    )

    view_spectrogram(iq_bb, fs_bb)


    # # 5. Detection & Final Output
    # viz.plot_spectrum(iq_bb, fs_bb, CONFIG["target_freq"], "2. Filtered Baseband Signal")
    # viz.plot_time(iq_bb, fs_bb, 0, 1, title="3. Baseband Chirp (Time Domain)")

    # detector = LoRaDetector(
    #     sf=CONFIG["lora_sf"], 
    #     bw=CONFIG["lora_bw"], 
    #     fs=fs_bb, 
    #     window_size=CONFIG["window_size"]
    # )
    
    # corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"])

    # plot_signal_quality(iq, fs_actual, detector.num_samples_per_symbol)
    # # plot_signal_quality(iq_bb, fs_bb, detector.num_samples_per_symbol)

    # if len(regions) > 0:
    #     display_title = meta.get('IQ_file', 'BANSHEE Capture')
    #     viz.plot_detections(
    #         corr,
    #         regions,
    #         fs_bb,
    #         headings_bb,
    #         title=f"BANSHEE Results: {display_title}"
    #     )
    # else:
    #     print(f"[!] No chirps detected at {CONFIG['target_freq']/1e6} MHz. Check threshold.")

    # lob_data = detector.get_robust_lob_region(regions)
    # if lob_data:
    #     # Get the heading at the exact center of the signal pulse
    #     target_heading_start = headings_bb[int(lob_data['start'])]
    #     target_heading_end = headings_bb[int(lob_data['stop'])]
    #     print(f"[*] Target Line of Bearing start: {target_heading_start:.2f} degrees")
    #     print(f"[*] Target Line of Bearing end: {target_heading_end:.2f} degrees")

if __name__ == "__main__":
    main()