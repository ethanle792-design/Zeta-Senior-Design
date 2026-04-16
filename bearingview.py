import numpy as np
import os
from IQ_Visualizer import extract_capture_metadata, IQDataManager
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector

# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "Apr8_Flight2/",
    "file_prefix": "spin_capture_", 
    "num_files": 7,                  
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.8,     
    "window_size": 1,
    "tests_per_file": 1,
}

def main():
    results_log = []

    # 1. Dynamic File Looping
    for file_num in range(CONFIG["num_files"]):
        meta_filename = f"{CONFIG['file_prefix']}{file_num}.json"
        meta_path = os.path.join(CONFIG["folder_path"], meta_filename)
        
        if not os.path.exists(meta_path):
            continue

        meta = extract_capture_metadata(meta_path)
        loader = IQDataManager(meta)

        iq_filename = os.path.join(CONFIG["folder_path"], meta.get('IQ_file', ''))
        compass_csv = os.path.join(CONFIG["folder_path"], meta.get('heading_file', ''))
        
        fs_actual = meta.get('rate')
        fc_actual = meta.get('freq')
        compass_offset = meta.get('log_delta_ns')

        if not os.path.exists(iq_filename):
            results_log.append(f"file {file_num}: ERROR (IQ file missing)")
            continue

        iq_full = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
        
        if os.path.exists(compass_csv):
            headings_full = loader.sync_sensors(compass_csv, len(iq_full), delay_offset=compass_offset * 1e-9)
        else:
            headings_full = np.full(len(iq_full), meta.get('heading_final', 0))

        # 2. Split File into Tests
        samples_per_test = len(iq_full) // CONFIG["tests_per_file"]

        for test_idx in range(CONFIG["tests_per_file"]):
            start_idx = test_idx * samples_per_test
            end_idx = (test_idx + 1) * samples_per_test
            
            iq = iq_full[start_idx:end_idx]
            headings = headings_full[start_idx:end_idx]
            
            # 3. DSP Pipeline
            sdc = SignalDownConverter(fs=fs_actual)
            freq_offset = CONFIG["target_freq"] - fc_actual
            iq_bb, fs_bb = sdc.process_pipeline(
                iq, 
                offset_hz=freq_offset, 
                bw_target=CONFIG["lora_bw"], 
                decimation_factor=CONFIG["decimation"]
            )
            headings_bb = headings[::CONFIG["decimation"]]

            # 4. Detection & Full Region Extraction
            detector = LoRaDetector(
                sf=CONFIG["lora_sf"], 
                bw=CONFIG["lora_bw"], 
                fs=fs_bb, 
                window_size=CONFIG["window_size"]
            )
            
            # Note: normalize=True used as requested for Apr8 data
            corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"], normalize=True)
            
            # Returns start/stop indices for the densest correlation cluster
            lob_data = detector.get_robust_lob_region(regions)
            
            if lob_data:
                # Map indices back to synchronized headings
                h_start = headings_bb[int(lob_data['start'])]
                h_stop = headings_bb[int(lob_data['stop'])]
                
                # Calculate the center of the detection window
                # We use modular arithmetic logic if your detector handles 360/0 wrap-around
                # but for simple index-based slicing:
                center_idx = int((lob_data['start'] + lob_data['stop']) / 2)
                h_center = headings_bb[center_idx]
                
                results_log.append(
                    f"file {file_num} test {test_idx + 1} heading: {h_center:.2f}° "
                    f"(Range: {h_start:.1f}° to {h_stop:.1f}°)"
                )
            else:
                results_log.append(f"file {file_num} test {test_idx + 1} heading: NO DETECTION")

    # Final Batch Print
    print("\n" + "="*50)
    print(f"BANSHEE BATCH REPORT - {CONFIG['folder_path']}")
    print("="*50)
    for entry in results_log:
        print(entry)
    print("="*50)

if __name__ == "__main__":
    main()