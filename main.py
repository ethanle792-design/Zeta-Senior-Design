import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager, plot_signal_quality, calculate_iq_weight
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram
from geospatial import SignalMapper

# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "Oval/",
    "file_prefix": "spin_capture_",  # Naming convention: spin_capture_0, spin_capture_1...
    "num_files": 7,                  # Set this to the [number] of files to process
    "bearing_offset_deg": -0.0,
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.8,     
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

            # 4. Detection & Robust LOB Extraction
            detector = LoRaDetector(sf=CONFIG["lora_sf"], bw=CONFIG["lora_bw"], fs=fs_bb, window_size=CONFIG["window_size"])
            corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"], normalize=True)
            
            # This identifies the densest region of chirps (the peak of your "spin")
            lob_data = detector.get_robust_lob_region(regions)
            
            if lob_data:
                # 1. Apply the calibration offset from CONFIG
                offset = CONFIG.get("bearing_offset_deg", 0.0)

                # 2. Extract and Rotate
                h_start = (headings_bb[int(lob_data['start'])] + offset) % 360
                h_stop = (headings_bb[int(lob_data['stop'])] + offset) % 360
                
                # Calculate center then apply offset (or vice versa, math is same)
                raw_center = headings_bb[int((lob_data['start'] + lob_data['stop']) / 2)]
                h_peak = (raw_center + offset) % 360
                
                estimate_entry = {
                    "source": meta_filename,
                    "test_id": f"{file_num}_{test_idx}",
                    "lat": lat,
                    "lon": lon,
                    "heading_start": h_start,
                    "heading_end": h_stop,
                    "heading_center": h_peak,
                    "max_corr": np.max(corr),
                    "weight": 1
                }
                all_estimates.append(estimate_entry)
                
                print(f"    [+] LOB Detected: {h_start:.1f}° to {h_stop:.1f}° (Peak: {h_peak:.1f}°)")

            # ==========================================
            # --- DEBUG PLOT GROUP ---
            # ==========================================
            if CONFIG["debug_plots"]:
                # --- Un-comment these as needed for deep-dives ---
                # viz.plot_constellation(iq)
                # viz.plot_histo(iq)
                # viz.plot_time(iq, fs_actual, 0, 1)
                
                viz.plot_spectrum(iq, fs_actual, fc_actual, f"Raw - File {file_num} Test {test_idx}")
                view_spectrogram(iq, fs_actual)
                viz.plot_spectrum(iq_bb, fs_bb, CONFIG["target_freq"], "Baseband")
                
                if len(regions) > 0:
                    viz.plot_detections(corr, regions, fs_bb, headings_bb, title=f"LOB Results - {meta_filename} [{test_idx}]")
                
                plot_signal_quality(iq, fs_actual, detector.num_samples_per_symbol)
                plt.show()

    print(f"\n[*] Batch processing complete. Total estimates saved: {len(all_estimates)}")
    return all_estimates

if __name__ == "__main__":
    # 1. Batch process the IQ files
    results = main() 
       # Test1 Apr8
    # true_lat, true_lon = 40.59084, -105.14109
        
        # Test2 apr8
    #true_lat, true_lon = 40.59147, -105.14148
    
    if results:
        mapper = SignalMapper(width=500, height=500, resolution=1)
        
        
        # Process data
        mapper.add_detections(results)
        
        # Optional: True location for validation
        
        # test1
        # true_loc = (40.59084, -105.14109)
        
        # test2
        # true_loc = (40.59147, -105.14148)
        
        # Oval
        true_loc = (40.576238, -105.081204)
        
        # IM
        # true_loc = (40.573389, -105.090778)
        
        # Visualize and Get Report
        fix, bbox = mapper.plot(true_coords=true_loc)

        print(f"\n--- SPECTRE REPORT ---")
        print(f"Estimated GPS: {fix['coords_gps']}")
        if bbox:
            print(f"Confidence Box: {bbox['width']:.1f}m x {bbox['height']:.1f}m")
            print(f"Search Area: {fix['search_area_m2']:.1f} m²")
            
