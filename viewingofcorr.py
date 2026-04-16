import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQDataManager
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector

# ==========================================
# --- MINIMAL CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "IM",
    "file_prefix": "spin_capture_",
    "num_files": 2,
    "tests_per_file": 1,
    "iq_format": "cs16",
    "target_freq": 915.5e6,
    "lora_sf": 12,
    "lora_bw": 125e3,
    "decimation": 4,
    "threshold": 0.8,
    "window_size": 1
}

def main():
    rows = CONFIG["num_files"]
    cols = CONFIG["tests_per_file"]
    # Added sharex=True to keep time scales aligned across subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows), squeeze=False, sharex=True)
    fig.suptitle(f"LoRa Matched Filter Correlation (Time Domain) - {CONFIG['folder_path']}", fontsize=16)

    for file_num in range(CONFIG["num_files"]):
        meta_filename = f"{CONFIG['file_prefix']}{file_num}.json"
        meta_path = os.path.join(CONFIG["folder_path"], meta_filename)
        
        if not os.path.exists(meta_path):
            continue

        meta = extract_capture_metadata(meta_path)
        loader = IQDataManager(meta)
        iq_filename = os.path.join(CONFIG["folder_path"], meta.get('IQ_file', ''))
        iq_full = loader.load_iq(iq_filename, format_type=CONFIG["iq_format"])
        
        samples_per_test = len(iq_full) // CONFIG["tests_per_file"]

        for test_idx in range(CONFIG["tests_per_file"]):
            ax = axes[file_num, test_idx]
            
            start_idx = test_idx * samples_per_test
            end_idx = (test_idx + 1) * samples_per_test
            iq = iq_full[start_idx:end_idx]

            # DSP Pipeline
            sdc = SignalDownConverter(fs=meta.get('rate'))
            freq_offset = CONFIG["target_freq"] - meta.get('freq')
            iq_bb, fs_bb = sdc.process_pipeline(
                iq, 
                offset_hz=freq_offset, 
                bw_target=CONFIG["lora_bw"], 
                decimation_factor=CONFIG["decimation"]
            )

            # 3. Correlation
            detector = LoRaDetector(
                sf=CONFIG["lora_sf"], 
                bw=CONFIG["lora_bw"], 
                fs=fs_bb, 
                window_size=CONFIG["window_size"]
            )
            corr, _ = detector.detect(iq_bb, threshold=CONFIG["threshold"], normalize=True)
            
            # --- TIME AXIS CALCULATION ---
            # Create a time array from 0 to (Length / Sample Rate)
            time_axis = np.arange(len(corr)) / fs_bb
            
            # 4. Plotting using the time_axis
            ax.plot(time_axis, corr, color='cyan', linewidth=0.8)
            ax.axhline(y=CONFIG["threshold"], color='red', linestyle='--', alpha=0.5)
            
            # Formatting
            if file_num == rows - 1: # Only label the bottom row
                ax.set_xlabel("Time (seconds)")
            
            if file_num == 0:
                ax.set_title(f"Test {test_idx}")
            if test_idx == 0:
                ax.set_ylabel(f"File {file_num}\nCorr Mag", rotation=90, size='small')
            
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()