import numpy as np
import scipy.signal as signal

class LoRaDetector:
    def __init__(self, sf, bw, fs):
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

# --- Integration Example ---
# Assuming 'metadata' is the dict from the JSON extraction function
# detector = LoRaDetector(
#     sf=10, 
#     bw=125000, 
#     fs=metadata['iq']['sample_rate']
# )

# iq_data = np.fromfile(metadata['iq']['filename'], dtype=np.complex64)
# correlation, detections = detector.detect(iq_data, threshold=0.6)