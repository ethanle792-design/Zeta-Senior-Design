import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from map import initialize_search_grid, extract_target_fix, plot_heatmap
from painting import paint_cone_to_grid, latlong_deg_to_meters

class SignalMapper:
    def __init__(self, width=500, height=500, resolution=1):
        self.width = width
        self.height = height
        self.res = resolution
        self.grid, self.meta = self._initialize_grid()
        self.anchor_lat = None
        self.anchor_lon = None
        self.drone_path_m = []
        
    def _initialize_grid(self):
        return initialize_search_grid(self.width, self.height, resolution_m=self.res)

    def set_anchor(self, lat, lon):
        self.anchor_lat = lat
        self.anchor_lon = lon

    def add_detections(self, results):
        if not self.anchor_lat:
            self.set_anchor(results[0]['lat'], results[0]['lon'])

        for detection in results:
            dx, dy = latlong_deg_to_meters(
                detection['lat'], detection['lon'], 
                self.anchor_lat, self.anchor_lon
            )
            self.drone_path_m.append((dx, dy))
            self.grid = paint_cone_to_grid(
                self.grid, self.meta, (dx, dy), 
                detection['heading_start'], detection['heading_end'],
                weight=detection.get('weight', 1)
            )

    def get_confidence_data(self, threshold_pct=0.95):
        """Calculates bbox and the geometric centroid of the confidence zone."""
        max_val = np.max(self.grid)
        if max_val <= 0: return None
        
        threshold = threshold_pct * max_val
        indices = np.argwhere(self.grid >= threshold)

        if indices.size == 0:
            return None

        # Convert grid indices to local meters
        # Note: indices are [row, col] -> [y, x]
        y_meters = self.meta['origin_y'] + indices[:, 0] * self.meta['res']
        x_meters = self.meta['origin_x'] + indices[:, 1] * self.meta['res']

        # Calculate Centroid (Mean of the confidence zone)
        centroid_x = np.mean(x_meters)
        centroid_y = np.mean(y_meters)

        return {
            'min_x': np.min(x_meters),
            'max_x': np.max(x_meters),
            'min_y': np.min(y_meters),
            'max_y': np.max(y_meters),
            'width': (np.max(x_meters) - np.min(x_meters)),
            'height': (np.max(y_meters) - np.min(y_meters)),
            'centroid': (centroid_x, centroid_y)
        }

    def plot(self, true_coords=None):
        """Renders heatmap with centroid-to-true-loc line and distance."""
        plot_heatmap(self.grid, self.meta, drone_path=self.drone_path_m)
        ax = plt.gca()

        # 1. Get Confidence Data
        conf_data = self.get_confidence_data()
        centroid_m = None

        if conf_data:
            # Draw Bounding Box
            rect = patches.Rectangle(
                (conf_data['min_x'], conf_data['min_y']), 
                conf_data['width'], conf_data['height'],
                linewidth=2, edgecolor='cyan', facecolor='none', 
                label=f'Confidence Zone', zorder=5
            )
            ax.add_patch(rect)
            
            # Plot Centroid (Center of the zone)
            centroid_m = conf_data['centroid']
            ax.scatter(centroid_m[0], centroid_m[1], c='blue', s=100, 
                       marker='o', label='Zone Centroid', zorder=6, edgecolors='white')

        # 2. True Location and Vector Logic
        if true_coords and self.anchor_lat:
            tx, ty = latlong_deg_to_meters(true_coords[0], true_coords[1], self.anchor_lat, self.anchor_lon)
            ax.scatter(tx, ty, c='lime', marker='X', s=200, label='True Loc', edgecolors='black', zorder=7)

            if centroid_m:
                # Calculate Euclidean Distance
                distance = np.sqrt((tx - centroid_m[0])**2 + (ty - centroid_m[1])**2)
                
                # Draw line from Centroid to True Loc
                ax.plot([centroid_m[0], tx], [centroid_m[1], ty], 
                        color='white', linestyle='--', linewidth=1.5, zorder=6)
                
                # Add distance text
                mid_x, mid_y = (centroid_m[0] + tx) / 2, (centroid_m[1] + ty) / 2
                ax.text(mid_x, mid_y, f'{distance:.2f}m', color='white', 
                        fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=2))
                
                print(f"Distance to target: {distance:.2f} meters")

        plt.legend(loc='upper right', fontsize='small')
        plt.show()