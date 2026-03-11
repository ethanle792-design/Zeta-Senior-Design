import numpy as np
import matplotlib.pyplot as plt
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector

# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "metadata_file": "cap_4_ethan.json",
    "iq_format": "cf32",         # 'cf32' or 'cs16'
    "target_freq": 915.4e6,      # Frequency of the LoRa channel
    "lora_sf": 12,               # Spreading Factor
    "lora_bw": 125e3,            # Bandwidth
    "decimation": 10,            # Downsample factor for processing speed
    "threshold": 0.7,            # Correlation peak threshold (0.0 to 1.0)
    "compass_offset": 0.45,      # Seconds compass started AFTER IQ log
    "debug_plots": False          # Toggle intermediate DSP plots
}

def main():
    # 0. Pre-run Cleanup
    plt.close('all')

    # 1. Initialization & Metadata
    print(f"[*] Initializing BANSHEE Pipeline for: {CONFIG['metadata_file']}")
    meta = extract_capture_metadata(CONFIG["metadata_file"])
    loader = IQDataManager(meta)
    viz = IQVisualizer(debug_mode=CONFIG["debug_plots"])

    # Extract hardware parameters from JSON
    fs_actual = meta['actual']['rate_sps']
    fc_actual = meta['actual']['freq_hz']
    iq_filename = meta['file_cf32']
    compass_csv = meta['sensors']['compass']['heading_csv']

    # 2. Data Loading & Sensor Sync
    print(f"[*] Loading IQ data from {iq_filename}...")
    iq = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
    
    # DEBUG 1: Raw Capture (Visual check of the noise floor/DC offset)
    viz.plot_spectrum(iq, fs_actual, fc_actual, "1. Raw Capture Spectrum")

    print("[*] Synchronizing Compass Data...")
    # FIX: Explicitly passing 'len(iq)' to satisfy the sync_sensors signature
    headings = loader.sync_sensors(
        compass_csv, 
        len(iq), 
        delay_offset=CONFIG["compass_offset"]
    )

    # 3. DSP Pipeline (DDC -> Filter -> Decimate)
    print(f"[*] Running Down-Conversion Pipeline...")
    sdc = SignalDownConverter(fs=fs_actual)
    
    # Shift target signal to 0 Hz and filter to target Bandwidth
    iq_bb, fs_bb = sdc.process_pipeline(
        iq, 
        offset_hz=(CONFIG["target_freq"] - fc_actual), 
        bw_target=CONFIG["lora_bw"], 
        decimation_factor=CONFIG["decimation"]
    )
    
    # Synchronize the heading array with the decimated IQ data
    headings_bb = headings[::CONFIG["decimation"]]

    # 4. Baseband Verification (Debug Plots)
    # DEBUG 2: Centered at 0Hz
    viz.plot_spectrum(iq_bb, fs_bb, CONFIG["target_freq"], "2. Filtered Baseband Signal")
    # DEBUG 3: Inspecting Chirp Slopes in Time
    viz.plot_time(iq_bb, fs_bb, num_samples=50000, title="3. Baseband Chirp (Time Domain)")

    # 5. Detection & Final Output
    print(f"[*] Detecting SF{CONFIG['lora_sf']} Chirps at {fs_bb/1e3:.1f} kSps...")
    detector = LoRaDetector(sf=CONFIG["lora_sf"], bw=CONFIG["lora_bw"], fs=fs_bb)
    corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"])

    # This call includes plt.show() as an anchor to hold all debug windows open
    if len(regions) > 0:
        viz.plot_detections(
            corr, 
            regions, 
            fs_bb, 
            headings_bb, 
            title=f"BANSHEE Results: {meta['base']} | Offset: {CONFIG['compass_offset']}s"
        )
    else:
        print("[!] No chirps detected above the threshold. Check frequency or SF settings.")

if __name__ == "__main__":
    main()