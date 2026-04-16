import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def paint_cone_to_grid(grid, meta, drone_m, start_deg, end_deg, weight=1.0):
    res = meta['res']
    X, Y = np.meshgrid(
        np.arange(meta['cols']) * res + meta['origin_x'],
        np.arange(meta['rows']) * res + meta['origin_y']
    )

    dx = X - drone_m[0]
    dy = Y - drone_m[1]
    angles_to_cells = np.degrees(np.arctan2(dx, dy)) % 360
    
    s = start_deg % 360
    e = end_deg % 360
    
    # CALCULATE ANGULAR WIDTH
    # This checks if we are painting the 'short' way or 'long' way
    diff = (e - s) % 360
    
    if diff <= 180:
        # Standard narrow cone
        if s <= e:
            inside_cone = (angles_to_cells >= s) & (angles_to_cells <= e)
        else:
            inside_cone = (angles_to_cells >= s) | (angles_to_cells <= e)
    else:
        # If the diff is > 180, the start/end were likely swapped.
        # We flip the logic to paint the small slice instead.
        if e <= s:
            inside_cone = (angles_to_cells >= e) & (angles_to_cells <= s)
        else:
            inside_cone = (angles_to_cells >= e) | (angles_to_cells <= s)
            
    grid[inside_cone] += weight
    return grid

def latlong_deg_to_meters(lat_deg, lon_deg, anchor_lat_deg, anchor_lon_deg):
    """Converts GPS (DEGREES) to meters relative to anchor"""
    R = 6378137.0 
    
    # Convert all inputs to radians for the math
    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    a_lat_rad, a_lon_rad = np.radians(anchor_lat_deg), np.radians(anchor_lon_deg)
    
    d_lat = lat_rad - a_lat_rad
    d_lon = lon_rad - a_lon_rad
    
    y = d_lat * R
    x = d_lon * R * np.cos(a_lat_rad)
    
    return x, y

import numpy as np


def paint_bearing_line(grid, lat, lon, h_peak, weight, anchor_lat, anchor_lon, meta):
    rows = meta['rows']
    cols = meta['cols']
    res = meta['res']
    
    # 1. Convert Lat/Lon to local meters (Standard BANSHEE logic)
    y_meters = (lat - anchor_lat) * 111111 
    x_meters = (lon - anchor_lon) * (111111 * np.cos(np.radians(anchor_lat)))
    
    # 2. Map meters to Grid Indices (Must match your pos_to_index logic)
    # Since origin is 'lower', we ADD the offset to move North/East
    y_drone = (y_meters / res) + (rows / 2)
    x_drone = (x_meters / res) + (cols / 2)
    
    # 3. Vector Direction (Natural Cartesian)
    # 0 deg (N) -> theta 90 -> dx=0, dy=1 (Up)
    # 180 deg (S) -> theta -90 -> dx=0, dy=-1 (Down)
    theta = np.radians(90 - h_peak)
    dx = np.cos(theta)
    dy = np.sin(theta) # Removed the negative sign!

    # 4. Trace the line
    max_len = int(np.sqrt(rows**2 + cols**2))
    for dist in range(max_len):
        curr_x = int(x_drone + dist * dx)
        curr_y = int(y_drone + dist * dy)
        
        if 0 <= curr_x < cols and 0 <= curr_y < rows:
            grid[curr_y, curr_x] += weight
            
    return grid