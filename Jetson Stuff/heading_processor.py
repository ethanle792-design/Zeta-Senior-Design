import numpy as np
import os
import sys
from IQ_Visualizer import extract_capture_metadata, IQDataManager, calculate_iq_weight
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector

# --- JETSON DEPLOYMENT CONFIG ---
CONFIG = {
    "bearing_offset_deg": 0.0,
    "iq_format": "cs16",         
    "target_freq": 915.5e6,      
    "lora_sf": 12,               
    "lora_bw": 125e3,            
    "decimation": 4,            
    "threshold": 0.8,     
    "window_size": 1
}

def process_single_recording(meta_path):
    """
    Processes a single IQ recording and returns the LOB estimate.
    Designed for headless execution on Jetson.
    """
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return None

    # 1. Load Metadata & Setup Loaders
    meta = extract_capture_metadata(meta_path)
    loader = IQDataManager(meta)
    
    # Resolve paths relative to the metadata file directory
    base_dir = os.path.dirname(meta_path)
    iq_filename = os.path.join(base_dir, meta.get('IQ_file', ''))
    compass_csv = os.path.join(base_dir, meta.get('heading_file', ''))
    
    lat = meta.get('lat')
    lon = meta.get('lon')
    fs_actual = meta.get('rate')
    fc_actual = meta.get('freq')
    compass_offset = meta.get('log_delta_ns', 0)

    if not os.path.exists(iq_filename):
        print(f"Error: IQ file {iq_filename} not found.")
        return None

    # 2. Load Data
    iq = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
    
    # Sync headings (using single static heading if CSV missing)
    if os.path.exists(compass_csv):
        headings = loader.sync_sensors(compass_csv, len(iq), delay_offset=compass_offset * 1e-9)
    else:
        headings = np.full(len(iq), meta.get('heading_final', 0))

    # 3. DSP Pipeline (Downconversion & Decimation)
    sdc = SignalDownConverter(fs=fs_actual)
    freq_offset = CONFIG["target_freq"] - fc_actual
    
    iq_bb, fs_bb = sdc.process_pipeline(
        iq, 
        offset_hz=freq_offset, 
        bw_target=CONFIG["lora_bw"], 
        decimation_factor=CONFIG["decimation"]
    )
    
    # Decimate headings to match IQ baseband
    headings_bb = headings[::CONFIG["decimation"]]

    # 4. Detection & LOB Extraction
    detector = LoRaDetector(
        sf=CONFIG["lora_sf"], 
        bw=CONFIG["lora_bw"], 
        fs=fs_bb, 
        window_size=CONFIG["window_size"]
    )
    
    corr, regions = detector.detect(iq_bb, threshold=CONFIG["threshold"], normalize=True)
    lob_data = detector.get_robust_lob_region(regions)
    
    if not lob_data:
        print("No LOB detected in this capture.")
        return None

    # 5. Apply Calibration & Package Estimate
    offset = CONFIG["bearing_offset_deg"]
    
    h_start = (headings_bb[int(lob_data['start'])] + offset) % 360
    h_stop = (headings_bb[int(lob_data['stop'])] + offset) % 360
    
    raw_center = headings_bb[int((lob_data['start'] + lob_data['stop']) / 2)]
    h_peak = (raw_center + offset) % 360
    
    estimate_dict = {
        "source": os.path.basename(meta_path),
        "lat": lat,
        "lon": lon,
        "heading_start": round(float(h_start), 2),
        "heading_end": round(float(h_stop), 2),
        "heading_center": round(float(h_peak), 2),
        "max_corr": round(float(np.max(corr)), 4),
        # "weight": calculate_iq_weight(iq) 
        "weight": 1
    }

    return estimate_dict

if __name__ == "__main__":
    # Example usage: python3 jetson_proc.py Oval/spin_capture_0.json
    if len(sys.argv) > 1:
        result = process_single_recording(sys.argv[1])
        if result:
            print(result)
    else:
        print("Usage: python3 script_name.py <path_to_json>")