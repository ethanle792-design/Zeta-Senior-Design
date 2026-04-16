import sys
import os

# Adds the parent directory (Zeta-Senior-Design) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager, plot_signal_quality, calculate_iq_weight
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram
from map import initialize_search_grid, extract_target_fix, plot_heatmap
from painting import paint_cone_to_grid, latlong_deg_to_meters
import matplotlib.patches as patches

# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "IM/",
    "file_prefix": "spin_capture_",  # Naming convention: spin_capture_0, spin_capture_1...
    "num_files": 2,                  # Set this to the [number] of files to process
    
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.34,     
    "window_size": 1,
    "tests_per_file": 1,
    "debug_plots": False         
}

def main():
    plt.close('all')
    all_estimates = []

    # 1. Dynamic File Looping
    for file_num in range(CONFIG["num_files"]):
        meta_filename = f"{CONFIG['file_prefix']}{file_num}.json"
        meta_path = os.path.join(CONFIG["folder_path"], meta_filename)
        
        if not os.path.exists(meta_path):
            print(f"[!] Skipping: {meta_path} (File not found)")
            continue

        print(f"\n[*] Processing File: {meta_path}")
        
        # Load metadata and setup loaders
        meta = extract_capture_metadata(meta_path)
        loader = IQDataManager(meta)
        viz = IQVisualizer(debug_mode=CONFIG["debug_plots"])

        iq_filename = os.path.join(CONFIG["folder_path"], meta.get('IQ_file', ''))
        compass_csv = os.path.join(CONFIG["folder_path"], meta.get('heading_file', ''))
        
        # Get GPS from metadata
        lat = meta.get('lat')
        lon = meta.get('lon')
        fs_actual = meta.get('rate')
        fc_actual = meta.get('freq')
        compass_offset = meta.get('log_delta_ns')

        if not os.path.exists(iq_filename):
            print(f"[!] ERROR: Could not find IQ file: {iq_filename}")
            continue

        # Load Full Data
        iq_full = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
        if os.path.exists(compass_csv):
            headings_full = loader.sync_sensors(compass_csv, len(iq_full), delay_offset=compass_offset * 1e-9)
        else:
            headings_full = np.full(len(iq_full), meta.get('heading_final', 0))

        # 2. Split File into 3 Tests
        samples_per_test = len(iq_full) // CONFIG["tests_per_file"]

        for test_idx in range(CONFIG["tests_per_file"]):
            print(f"  [>] Test {test_idx + 1}/{CONFIG['tests_per_file']}")
            
            start_idx = test_idx * samples_per_test
            end_idx = (test_idx + 1) * samples_per_test
            
            iq = iq_full[start_idx:end_idx]
            headings = headings_full[start_idx:end_idx]
            
            weight = calculate_iq_weight(iq)

            # 3. DSP Pipeline
            sdc = SignalDownConverter(fs=fs_actual)
            freq_offset = CONFIG["target_freq"] - fc_actual
            iq_bb, fs_bb = sdc.process_pipeline(iq, offset_hz=freq_offset, bw_target=CONFIG["lora_bw"], decimation_factor=CONFIG["decimation"])
            headings_bb = headings[::CONFIG["decimation"]]
            
            time_start = 4
            time_end = 4.5
            start = int(time_start * fs_bb)
            stop =  int(time_end * fs_bb)
            print(f"start:", start)
            print(f"stop:", stop)
            print(f"length:", len(iq_bb))
            #view_spectrogram(iq_bb[start:stop], fs_bb)
            view_spectrogram(iq_bb, fs_bb)

            
    return all_estimates

if __name__ == "__main__":
    # 1. Batch process the IQ files
    results = main() 
    
   