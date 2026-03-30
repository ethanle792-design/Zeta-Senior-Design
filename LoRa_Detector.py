import numpy as np
import scipy.signal as signal

class LoRaDetector:
    def __init__(self, sf, bw, fs, window_size):
        """
        Initialize the detector with fixed radio parameters.
        
        Args:
            sf (int): Spreading Factor (7-12)
            bw (float): Bandwidth in Hz
            fs (float): Sample Rate in Hz
        """
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.window_size = window_size
        
        # Pre-calculate symbol duration and reference chirp
        self.num_samples_per_symbol = int((2**sf / bw) * fs)
        self.reference_chirp = self._generate_ideal_upchirp()
        
        # Pre-calculate the Matched Filter kernel (Conjugate + Time Reversal)
        self.mf_kernel = np.conj(self.reference_chirp[::-1])

    def _generate_ideal_upchirp(self):
        """Generates a standard LoRa 'Upchirp' centered at baseband."""
        t = np.arange(self.num_samples_per_symbol) / self.fs
        k = self.bw / (self.num_samples_per_symbol / self.fs)
        # Sweep from -BW/2 to +BW/2
        chirp = np.exp(1j * 2 * np.pi * ((-self.bw/2) * t + 0.5 * k * t**2))
        return chirp.astype(np.complex64)

    def detect(self, rx_signal, threshold=0.5, normalize=True):
        """
        Runs the matched filter and finds regions of interest.
        
        Returns:
            corr (np.array): The correlation magnitude
            regions (list): List of (start, stop) tuples for detected chirps
        """
        # 1. Fast FFT Convolution
        corr_full = signal.fftconvolve(rx_signal, self.mf_kernel, mode='full')
        
        # 2. Align output to input (Trim convolution tail)
        # The peak occurs at the end of the matched pattern
        corr = corr_full[self.num_samples_per_symbol - 1 : 
                         self.num_samples_per_symbol - 1 + len(rx_signal)]
        
        corr_mag = np.abs(corr)
        
        if normalize:
            corr_mag /= (np.max(corr_mag) + 1e-12)

        # 3. Extract Regions
        corr_mag = self.moving_average(corr_mag)
        regions = self._get_regions(corr_mag, threshold)
        
        return corr_mag, regions

    def _get_regions(self, corr, threshold):
        """Finds contiguous regions above the threshold."""
        above = corr >= threshold
        edges = np.diff(above.astype(int))
        
        starts = np.where(edges == 1)[0] + 1
        stops = np.where(edges == -1)[0] + 1

        if above[0]: starts = np.insert(starts, 0, 0)
        if above[-1]: stops = np.append(stops, len(corr))

        return list(zip(starts, stops))
    
    def moving_average(self, corr):
        corr = np.asarray(corr)
        window = np.ones(self.window_size) / self.window_size
        return np.convolve(corr, window, mode="same")

    def get_robust_lob_region(self, regions, max_gap_samples=None):
        if not regions or len(regions) < 3:
            return None

        # 1. Automatic Gap Threshold (if not provided)
        # Default: 2x the expected chirp duration
        if max_gap_samples is None:
            expected_chirp = (2**self.sf / self.bw) * self.fs
            max_gap_samples = expected_chirp * 10

        # 2. Clustering: Group regions that are 'close' in time
        clusters = []
        current_cluster = [regions[0]]

        for i in range(1, len(regions)):
            gap = regions[i][0] - regions[i-1][1]
            if gap < max_gap_samples:
                current_cluster.append(regions[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [regions[i]]
        clusters.append(current_cluster)

        # 3. Filter and Score Clusters
        # We want the cluster with the most detections (highest density)
        # This ignores random noise spikes elsewhere in the capture.
        best_cluster = max(clusters, key=len)
        
        if len(best_cluster) < 5: # Minimum detections to be a 'packet'
            print("[!] No dense clusters found. Likely noise.")
            return None

        # 4. Calculate Metrics for the Line of Bearing
        starts = [r[0] for r in best_cluster]
        stops = [r[1] for r in best_cluster]
        
        global_start = starts[0]
        global_stop = stops[-1]
        
        # Temporal Centroid: The 'average' time of the packet
        # Midpoints of every chirp averaged together
        midpoints = [(s + e) / 2 for s, e in best_cluster]
        centroid_sample = np.mean(midpoints)

        return {
            "start": global_start,
            "stop": global_stop,
            "centroid": centroid_sample,
            "count": len(best_cluster),
            "density": len(best_cluster) / (global_stop - global_start)
        }

# --- Integration Example ---
# Assuming 'metadata' is the dict from the JSON extraction function
# detector = LoRaDetector(
#     sf=10, 
#     bw=125000, 
#     fs=metadata['iq']['sample_rate']
# )

# iq_data = np.fromfile(metadata['iq']['filename'], dtype=np.complex64)
# correlation, detections = detector.detect(iq_data, threshold=0.6)