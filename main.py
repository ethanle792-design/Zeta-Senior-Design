import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram

# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "metadata_file": "Mar27_Field/Site2_spin.json",
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.350,     
    "window_size": 5,         
    "debug_plots": False         
}

def main():
    plt.close('all')

    # 1. Initialization & Metadata
    print(f"[*] Initializing BANSHEE Pipeline for: {CONFIG['metadata_file']}")
    
    # Get the directory where the JSON lives to build absolute paths
    base_dir = os.path.dirname(CONFIG["metadata_file"])
    
    # Load the JSON content
    meta = extract_capture_metadata(CONFIG["metadata_file"])
    loader = IQDataManager(meta)
    viz = IQVisualizer(debug_mode=CONFIG["debug_plots"])

    # 2. Path & Key Resolution (Updated to match your JSON keys)
    # Using 'IQ_file' and 'heading_file' directly from your JSON
    iq_filename = os.path.join(base_dir, meta.get('IQ_file', ''))
    compass_csv = os.path.join(base_dir, meta.get('heading_file', ''))
    
    fs_actual = meta.get('rate')
    fc_actual = meta.get('freq')
    
    compass_offset = meta.get('log_delta_ns')

    # 3. Data Loading & Sensor Sync
    print(f"[*] Loading IQ data from {iq_filename}...")
    if not os.path.exists(iq_filename):
        print(f"[!] ERROR: Could not find IQ file at {iq_filename}")
        return

    iq = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
    
    viz.plot_spectrum(iq, fs_actual, fc_actual, "1. Raw Capture Spectrum")
    
    #view_spectrogram(iq, fs_actual)

    print(f"[*] Synchronizing Compass Data from {compass_csv}...")
    if os.path.exists(compass_csv):
        headings = loader.sync_sensors(
            compass_csv, 
            len(iq), 
            delay_offset= compass_offset * 10e-9
        )
    else:
        # Fallback to 'heading_final' if the CSV is missing
        final_h = meta.get('heading_final', 0)
        print(f"[!] Warning: {compass_csv} not found. Using final heading: {final_h}")
        headings = np.full(len(iq), final_h)

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
    
    # Downsample headings to match decimated IQ length
    headings_bb = headings[::CONFIG["decimation"]]

    # 5. Detection & Final Output
    viz.plot_spectrum(iq_bb, fs_bb, CONFIG["target_freq"], "2. Filtered Baseband Signal")
    viz.plot_time(iq_bb, fs_bb, num_samples=50000, title="3. Baseband Chirp (Time Domain)")

    detector = LoRaDetector(sf=CONFIG["lora_sf"], bw=CONFIG["lora_bw"], fs=fs_bb, window_size=CONFIG["window_size"])
    corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"])

    if len(regions) > 0:
        display_title = meta.get('IQ_file', 'BANSHEE Capture')
        viz.plot_detections(
            corr, 
            regions, 
            fs_bb, 
            headings_bb, 
            title=f"BANSHEE Results: {display_title}"
        )
    else:
        print(f"[!] No chirps detected at {CONFIG['target_freq']/1e6} MHz. Check threshold.")
        
    lob_data = detector.get_robust_lob_region(regions)
    if lob_data:
        # Get the heading at the exact center of the signal pulse
        target_heading_start = headings_bb[int(lob_data['start'])]
        target_heading_end = headings_bb[int(lob_data['stop'])]
        print(f"[*] Target Line of Bearing start: {target_heading_start:.2f} degrees")
        print(f"[*] Target Line of Bearing end: {target_heading_end:.2f} degrees")

if __name__ == "__main__":
    main()