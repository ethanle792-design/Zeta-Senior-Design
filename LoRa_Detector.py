import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class LoRaDetector:
    def __init__(self, sf: int, bw: float, fs: float, window_size: int):
        """
        Initialize the detector with fixed radio parameters.

        Args:
            sf:          Spreading factor (7–12).
            bw:          Bandwidth (Hz).
            fs:          Sample rate (Hz).
            window_size: Moving-average window length for correlation smoothing.
        """
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.window_size = window_size

        # Pre-calculate symbol duration and reference chirp
        self.num_samples_per_symbol = int((2**sf / bw) * fs)
        self.reference_chirp = self._generate_ideal_upchirp()

        # Matched filter kernel: conjugate + time reversal of reference
        self.mf_kernel = np.conj(self.reference_chirp[::-1])

    # ==========================================
    # --- SIGNAL GENERATION ---
    # ==========================================

    def _generate_ideal_upchirp(self) -> np.ndarray:
        """
        Generates a standard LoRa upchirp centered at baseband.
        Sweeps from -BW/2 to +BW/2 over one symbol duration.
        """
        t = np.arange(self.num_samples_per_symbol) / self.fs
        k = self.bw / (self.num_samples_per_symbol / self.fs)
        chirp = np.exp(1j * 2 * np.pi * ((-self.bw / 2) * t + 0.5 * k * t**2))
        return chirp.astype(np.complex64)

    # ==========================================
    # --- DETECTION PIPELINE ---
    # ==========================================

    def detect(self, rx_signal: np.ndarray,
               threshold: float = 0.5,
               normalize: bool = True,
               debug: bool = False) -> tuple[np.ndarray, list]:
        """
        Runs the matched filter and extracts regions of interest.

        Pipeline: FFT convolution → alignment → magnitude → 
                  normalization → moving average → region extraction.

        Args:
            rx_signal: Baseband complex IQ samples.
            threshold: Detection threshold (0–1 if normalized).
            normalize: Normalize correlation output to [0, 1].
            debug:     If True, plots every intermediate stage.

        Returns:
            corr_mag: Smoothed, (optionally normalized) correlation magnitude.
            regions:  List of (start, stop) sample index tuples for detections.
        """

        # --- Stage 1: Reference chirp ---
        if debug:
            self._plot_reference_chirp()

        # --- Stage 2: FFT convolution ---
        # Equivalent to sliding the MF kernel across the received signal.
        # Output length = len(rx) + len(kernel) - 1; we trim below.
        corr_full = signal.fftconvolve(rx_signal, self.mf_kernel, mode='full')

        if debug:
            self._plot_raw_correlation(corr_full)

        # --- Stage 3: Alignment ---
        # Peak occurs at sample (num_samples_per_symbol - 1) into the output.
        # Trim to align correlation output with the input sample timeline.
        corr = corr_full[self.num_samples_per_symbol - 1:
                         self.num_samples_per_symbol - 1 + len(rx_signal)]
        corr_mag = np.abs(corr)

        if debug:
            self._plot_aligned_correlation(corr_mag, label="Stage 3: Aligned Correlation Magnitude (pre-norm)")

        # --- Stage 4: Normalization ---
        if normalize:
            corr_mag /= (np.max(corr_mag) + 1e-12)

        if debug:
            self._plot_aligned_correlation(corr_mag, label="Stage 4: Normalized Correlation [0, 1]",
                                           threshold=threshold)

        # --- Stage 5: Moving average smoothing ---
        corr_mag = self.moving_average(corr_mag)

        if debug:
            self._plot_aligned_correlation(corr_mag,
                                           label=f"Stage 5: Smoothed (window={self.window_size})",
                                           threshold=threshold)

        # --- Stage 6: Region extraction ---
        regions = self._get_regions(corr_mag, threshold)

        if debug:
            self._plot_regions(corr_mag, regions, threshold)
            # plt.show() intentionally omitted — caller owns the show call

        return corr_mag, regions

    def get_robust_lob_region(self, regions: list,
                              max_gap_samples: float = None,
                              debug: bool = False) -> dict | None:
        """
        Finds the densest cluster of detections — the peak of the spin arc.

        Clusters nearby regions, scores by detection count, and returns
        the temporal centroid and span of the best cluster.

        Args:
            regions:          List of (start, stop) tuples from detect().
            max_gap_samples:  Max inter-region gap to consider same cluster.
                              Defaults to 10× the expected chirp duration.
            debug:            If True, plots clustering result.

        Returns:
            Dict with start, stop, centroid, count, density — or None if
            no cluster meets the minimum detection count.
        """
        if not regions or len(regions) < 3:
            print("[!] Fewer than 3 regions detected — likely noise or no signal.")
            return None

        # --- Auto gap threshold: 10× expected chirp duration ---
        if max_gap_samples is None:
            expected_chirp = (2**self.sf / self.bw) * self.fs
            max_gap_samples = expected_chirp * 10

        # --- Cluster regions that are close in time ---
        clusters = []
        current_cluster = [regions[0]]

        for i in range(1, len(regions)):
            gap = regions[i][0] - regions[i - 1][1]
            if gap < max_gap_samples:
                current_cluster.append(regions[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [regions[i]]
        clusters.append(current_cluster)

        # --- Score: largest cluster wins ---
        best_cluster = max(clusters, key=len)

        if len(best_cluster) < 5:
            print("[!] No dense cluster found — likely noise.")
            return None

        starts = [r[0] for r in best_cluster]
        stops = [r[1] for r in best_cluster]

        global_start = starts[0]
        global_stop = stops[-1]

        # Temporal centroid: mean midpoint of all chirps in the cluster
        midpoints = [(s + e) / 2 for s, e in best_cluster]
        centroid_sample = np.mean(midpoints)

        lob_data = {
            "start":    global_start,
            "stop":     global_stop,
            "centroid": centroid_sample,
            "count":    len(best_cluster),
            "density":  len(best_cluster) / (global_stop - global_start)
        }

        if debug:
            self._plot_lob_result(clusters, best_cluster, lob_data)
            # plt.show() intentionally omitted — caller owns the show call

        return lob_data

    # ==========================================
    # --- DSP HELPERS ---
    # ==========================================

    def _get_regions(self, corr: np.ndarray, threshold: float) -> list:
        """Finds contiguous sample runs above the threshold."""
        above = corr >= threshold
        edges = np.diff(above.astype(int))

        starts = np.where(edges == 1)[0] + 1
        stops = np.where(edges == -1)[0] + 1

        if above[0]:
            starts = np.insert(starts, 0, 0)
        if above[-1]:
            stops = np.append(stops, len(corr))

        return list(zip(starts, stops))

    def moving_average(self, corr: np.ndarray) -> np.ndarray:
        corr = np.asarray(corr)
        window = np.ones(self.window_size) / self.window_size
        return np.convolve(corr, window, mode='same')

    # ==========================================
    # --- DEBUG PLOTS ---
    # ==========================================

    def _plot_reference_chirp(self) -> None:
        """Plots the reference upchirp: spectrogram, I/Q time series, and constellation."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Stage 1: Reference Upchirp", fontsize=12, fontweight='bold')
        fig.text(0.5, 0.92,
                 f"SF={self.sf}  |  BW={self.bw/1e3:.1f} kHz  |  "
                 f"fs={self.fs/1e3:.1f} kHz  |  {self.num_samples_per_symbol} samples/symbol",
                 ha='center', fontsize=9, color='gray')

        t_us = np.arange(self.num_samples_per_symbol) / self.fs * 1e6

        # Spectrogram
        axes[0].specgram(self.reference_chirp, Fs=self.fs, cmap='viridis')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_title("Spectrogram")

        # I/Q time series
        axes[1].plot(t_us, np.real(self.reference_chirp), linewidth=0.7,
                     label='I', color='steelblue')
        axes[1].plot(t_us, np.imag(self.reference_chirp), linewidth=0.7,
                     label='Q', color='coral', alpha=0.8)
        axes[1].set_xlabel("Time (µs)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("I/Q Time Series")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Constellation
        axes[2].scatter(np.real(self.reference_chirp), np.imag(self.reference_chirp),
                        s=1, alpha=0.5, color='steelblue')
        axes[2].set_xlabel("I")
        axes[2].set_ylabel("Q")
        axes[2].set_title("Constellation")
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='gray', linewidth=0.5)
        axes[2].axvline(0, color='gray', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.90])

    def _plot_raw_correlation(self, corr_full: np.ndarray) -> None:
        """Plots the raw FFT convolution output (full length, pre-alignment)."""
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle("Stage 2: Raw FFT Convolution Output (pre-alignment)",
                     fontsize=12, fontweight='bold')
        fig.text(0.5, 0.92,
                 f"Length: {len(corr_full)} samples  |  "
                 f"Tail of {self.num_samples_per_symbol - 1} samples will be trimmed",
                 ha='center', fontsize=9, color='gray')

        ax.plot(np.abs(corr_full), linewidth=0.6, color='steelblue')

        # Mark the trim boundary
        ax.axvline(self.num_samples_per_symbol - 1, color='coral',
                   linestyle='--', linewidth=1, label='Alignment start')
        ax.axvline(self.num_samples_per_symbol - 1 + (len(corr_full) - self.num_samples_per_symbol + 1),
                   color='orange', linestyle='--', linewidth=1, label='Alignment end')

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Correlation magnitude")
        ax.set_title("Full Convolution Output")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.90])

    def _plot_aligned_correlation(self, corr_mag: np.ndarray,
                                  label: str = "",
                                  threshold: float = None) -> None:
        """Plots correlation magnitude aligned to the input sample timeline."""
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle(label, fontsize=12, fontweight='bold')

        ax.plot(corr_mag, linewidth=0.6, color='steelblue')

        if threshold is not None:
            ax.axhline(threshold, color='coral', linestyle='--',
                       linewidth=1, label=f'Threshold = {threshold}')
            ax.legend(fontsize=8)

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def _plot_regions(self, corr_mag: np.ndarray,
                      regions: list, threshold: float) -> None:
        """Plots the final smoothed correlation with detected regions shaded."""
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle("Stage 6: Detected Regions", fontsize=12, fontweight='bold')
        fig.text(0.5, 0.92,
                 f"{len(regions)} region(s) detected above threshold={threshold}",
                 ha='center', fontsize=9, color='gray')

        ax.plot(corr_mag, linewidth=0.6, color='steelblue', zorder=2)
        ax.axhline(threshold, color='coral', linestyle='--',
                   linewidth=1, label=f'Threshold = {threshold}')

        for i, (start, stop) in enumerate(regions):
            ax.axvspan(start, stop, alpha=0.25, color='green',
                       label='Detected region' if i == 0 else None)

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Normalized magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.90])

    def _plot_lob_result(self, clusters: list, best_cluster: list,
                         lob_data: dict) -> None:
        """
        Plots cluster scoring and the final LOB selection.
        Left panel: bar chart of cluster sizes.
        Right panel: timeline showing all clusters, with the best highlighted.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LOB Clustering Result", fontsize=12, fontweight='bold')
        fig.text(0.5, 0.92,
                 f"Best cluster: {lob_data['count']} detections  |  "
                 f"span: {lob_data['start']}–{lob_data['stop']}  |  "
                 f"centroid: {lob_data['centroid']:.0f}",
                 ha='center', fontsize=9, color='gray')

        # --- Cluster size bar chart ---
        ax = axes[0]
        sizes = [len(c) for c in clusters]
        colors = ['steelblue'] * len(clusters)
        best_idx = clusters.index(best_cluster)
        colors[best_idx] = 'coral'

        ax.bar(range(len(clusters)), sizes, color=colors)
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Detection count")
        ax.set_title("Cluster Sizes (coral = selected)")
        ax.grid(True, alpha=0.3, axis='y')

        # --- Timeline ---
        ax = axes[1]
        for c_idx, cluster in enumerate(clusters):
            color = 'coral' if cluster is best_cluster else 'steelblue'
            alpha = 0.8 if cluster is best_cluster else 0.3
            for start, stop in cluster:
                ax.axvspan(start, stop, ymin=0.1, ymax=0.9,
                           alpha=alpha, color=color)

        # Mark centroid
        ax.axvline(lob_data['centroid'], color='black', linestyle='--',
                   linewidth=1.2, label=f"Centroid: {lob_data['centroid']:.0f}")
        ax.axvline(lob_data['start'], color='green', linestyle=':',
                   linewidth=1, label=f"LOB start: {lob_data['start']}")
        ax.axvline(lob_data['stop'], color='green', linestyle=':',
                   linewidth=1, label=f"LOB stop: {lob_data['stop']}")

        ax.set_xlabel("Sample index")
        ax.set_title("Detection Timeline (coral = selected cluster)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.90])