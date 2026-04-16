import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQDataManager, calculate_iq_weight

CONFIG = {
    "folder_path": "Apr8_Flight1/",
    "file_prefix": "spin_capture_",
    "num_files": 3,          # Total files to view
    "tests_per_file": 3,     # Slices per file
    "iq_format": "cs16",
}

def main():
    plt.close('all')
    
    # Calculate grid dimensions for subplots
    rows = CONFIG["num_files"]
    cols = CONFIG["tests_per_file"]
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    
    # Ensure axes is a 2D array even if rows=1
    if rows == 1: axes = np.expand_dims(axes, axis=0)

    for r in range(rows):
        meta_path = os.path.join(CONFIG["folder_path"], f"{CONFIG['file_prefix']}{r}.json")
        if not os.path.exists(meta_path): continue

        meta = extract_capture_metadata(meta_path)
        loader = IQDataManager(meta)
        iq_file = os.path.join(CONFIG["folder_path"], meta.get('IQ_file', ''))
        iq_full = loader.load_iq(iq_file, format_type=CONFIG["iq_format"])
        
        samples_per_test = len(iq_full) // cols

        for c in range(cols):
            start, end = c * samples_per_test, (c + 1) * samples_per_test
            iq = iq_full[start:end]
            
            # Calculate quality weight
            weight = calculate_iq_weight(iq)
            
            ax = axes[r, c]
            # Density plot (Hexbin)
            hb = ax.hexbin(iq.real, iq.imag, gridsize=70, cmap='magma', bins='log')
            
            # Formatting
            # ax.set_title(f"File {r} | Test {c+1}\nWeight: {weight:.2f}")
            ax.set_title(f"File {r} | Test {c+1}\n")
            ax.set_facecolor('black')
            ax.set_aspect('equal')
            
            # Dynamic scaling based on signal strength
            limit = np.max(np.abs(iq)) * 1.1 if len(iq) > 0 else 1
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            
            # Hide ticks for a cleaner look
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"BANSHEE IQ Constellation Batch: {CONFIG['folder_path']}", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()