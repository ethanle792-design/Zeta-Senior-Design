import numpy as np
import matplotlib.pyplot as plt

def initialize_search_grid(width_m, height_m, resolution_m=1.0):
    rows = int(height_m / resolution_m)
    cols = int(width_m / resolution_m)
    grid = np.zeros((rows, cols), dtype=np.float32)
    
    meta = {
        'res': resolution_m,
        'width_m': width_m,
        'height_m': height_m,
        'origin_x': -width_m / 2, # Center-anchored
        'origin_y': -height_m / 2, 
        'rows': rows,
        'cols': cols
    }
    print(f"Grid Initialized: {rows}x{cols} cells ({resolution_m}m resolution)")
    return grid, meta

def pos_to_index(x, y, meta):
    col = int((x - meta['origin_x']) / meta['res'])
    row = int((y - meta['origin_y']) / meta['res'])
    if 0 <= col < meta['cols'] and 0 <= row < meta['rows']:
        return row, col
    return None

def plot_heatmap(grid, meta, drone_path=None):
    plt.figure(figsize=(8, 12))
    extent = [meta['origin_x'], meta['origin_x'] + meta['width_m'],
              meta['origin_y'], meta['origin_y'] + meta['height_m']]
    
    plt.imshow(grid, origin='lower', extent=extent, cmap='magma')
    plt.colorbar(label='Detection Confidence (Votes)')
    
    if drone_path:
        path_x, path_y = zip(*drone_path)
        plt.plot(path_x, path_y, 'c--', alpha=0.6, label='Drone Path')
    
    plt.title("BANSHEE Probabilistic Heatmap")
    plt.xlabel("Meters East")
    plt.ylabel("Meters North")

def extract_target_fix(grid, meta, anchor_gps_deg):
    """Finds peak, 90% confidence area, and GPS coordinates."""
    # 1. Find Peak
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    max_val = np.max(grid)
    
    # 2. Convert to Meters
    est_x = meta['origin_x'] + (idx[1] * meta['res'])
    est_y = meta['origin_y'] + (idx[0] * meta['res'])
    
    # 3. Convert to Decimal Degrees
    R = 6378137.0 
    a_lat, a_lon = anchor_gps_deg
    est_lat = a_lat + np.degrees(est_y / R)
    est_lon = a_lon + np.degrees(est_x / (R * np.cos(np.radians(a_lat))))

    # 4. Search Area (Confidence Metric)
    threshold = 0.9 * max_val
    probable_cells = np.sum(grid >= threshold)
    area_m2 = probable_cells * (meta['res']**2)

    return {
        'coords_m': (est_x, est_y),
        'coords_gps': (est_lat, est_lon),
        'confidence_peak': max_val,
        'search_area_m2': area_m2
    }