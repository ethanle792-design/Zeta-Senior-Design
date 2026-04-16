import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager, plot_signal_quality, calculate_iq_weight
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram
from geospatial import SignalMapper

# ==========================================
# --- SINGLE FILE CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "Apr8_Flight2/",
    "file_name": "spin_capture_3.json", # Point directly to your target JSON
    "bearing_offset_deg": 0.0,
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.8,     
    "window_size": 1,
    "debug_plots": True # Enabled for viewing debug plots
}

def main():
    plt.close('all')
    
    meta_path = os.path.join(CONFIG["folder_path"], CONFIG["file_name"])
    
    if not os.path.exists(meta_path):
        print(f"[!] ERROR: File not found: {meta_path}")
        return

    print(f"[*] Processing Single File: {meta_path}")
    
    # 1. Load Metadata and Setup
    meta = extract_capture_metadata(meta_path)
    loader = IQDataManager(meta)
    viz = IQVisualizer(debug_mode=CONFIG["debug_plots"])

    iq_filename = os.path.join(CONFIG["folder_path"], meta.get('IQ_file', ''))
    compass_csv = os.path.join(CONFIG["folder_path"], meta.get('heading_file', ''))
    
    lat = meta.get('lat')
    lon = meta.get('lon')
    fs_actual = meta.get('rate')
    fc_actual = meta.get('freq')
    compass_offset = meta.get('log_delta_ns')

    # 2. Load IQ and Sync Headings
    iq = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
    if os.path.exists(compass_csv):
        headings = loader.sync_sensors(compass_csv, len(iq), delay_offset=compass_offset * 1e-9)
    else:
        headings = np.full(len(iq), meta.get('heading_final', 0))

    weight = calculate_iq_weight(iq)

    # 3. DSP Pipeline
    sdc = SignalDownConverter(fs=fs_actual)
    freq_offset = CONFIG["target_freq"] - fc_actual
    iq_bb, fs_bb = sdc.process_pipeline(iq, offset_hz=freq_offset, bw_target=CONFIG["lora_bw"], decimation_factor=CONFIG["decimation"])
    headings_bb = headings[::CONFIG["decimation"]]

    # 4. Detection & LOB Extraction
    detector = LoRaDetector(sf=CONFIG["lora_sf"], bw=CONFIG["lora_bw"], fs=fs_bb, window_size=CONFIG["window_size"])
    corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"], normalize=True)
    
    lob_data = detector.get_robust_lob_region(regions)
    estimate_entry = None

    if lob_data:
        offset = CONFIG["bearing_offset_deg"]
        h_start = (headings_bb[int(lob_data['start'])] + offset) % 360
        h_stop = (headings_bb[int(lob_data['stop'])] + offset) % 360
        h_peak = (headings_bb[int((lob_data['start'] + lob_data['stop']) / 2)] + offset) % 360
        
        estimate_entry = {
            "source": CONFIG["file_name"],
            "test_id": "single_run",
            "lat": lat,
            "lon": lon,
            "heading_start": h_start,
            "heading_end": h_stop,
            "heading_center": h_peak,
            "max_corr": np.max(corr),
            "weight": weight
        }
        print(f"    [+] LOB Detected: {h_start:.1f}° to {h_stop:.1f}° (Peak: {h_peak:.1f}°)")

    # 5. DEBUG PLOTS
    if CONFIG["debug_plots"]:
        print("[*] Generating debug plots...")
        viz.plot_spectrum(iq, fs_actual, fc_actual, f"Raw Spectrum: {CONFIG['file_name']}")
        view_spectrogram(iq[:len(iq)//72], fs_actual)
        viz.plot_spectrum(iq_bb, fs_bb, CONFIG["target_freq"], "Downconverted Baseband")
        
        if len(regions) > 0:
            viz.plot_detections(corr, regions, fs_bb, headings_bb, title=f"LOB Results: {CONFIG['file_name']}")
        
        plot_signal_quality(iq, fs_actual, detector.num_samples_per_symbol)
        
        # This keeps the windows open until you close them manually
        plt.show()

    return [estimate_entry] if estimate_entry else []

if __name__ == "__main__":
    results = main() 
    
    if results:
        # Run the mapper for the single detection
        mapper = SignalMapper(width=500, height=500, resolution=1)
        mapper.add_detections(results)
        
        # Update these coordinates if needed for your specific capture site
        true_loc = (40.59147, -105.14148)
        fix, bbox = mapper.plot(true_coords=true_loc)

        print(f"\n--- SINGLE FILE ANALYSIS COMPLETE ---")
        print(f"Estimated GPS: {fix['coords_gps']}")
        plt.show()