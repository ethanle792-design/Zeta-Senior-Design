import numpy as np
import matplotlib.pyplot as plt
import os
from IQ_Visualizer import extract_capture_metadata, IQVisualizer, IQDataManager, plot_signal_quality, calculate_iq_weight
from Signal_Downconverter import SignalDownConverter
from LoRa_Detector import LoRaDetector
from Spectrogram import view_spectrogram
from geospatial import SignalMapper


# ==========================================
# --- SIGNAL & HARDWARE CONFIGURATION ---
# ==========================================
CONFIG = {
    "folder_path": "Oval/",
    "file_prefix": "spin_capture_",  # Naming convention: spin_capture_0, spin_capture_1...
    "num_files": 21,                  # Number of files to process in batch mode
    "bearing_offset_deg": -0.0,
    "iq_format": "cs16",
    "target_freq": 915.5e6,
    "lora_sf": 12,
    "lora_bw": 125e3,
    "decimation": 4,
    "threshold": 0.8,
    "window_size": 1,
    "tests_per_file": 1,

    # ---- Debug mode ----
    # When debug_plots=True the processor ignores num_files and processes only
    # the single file specified by debug_file_num. Only the first debug_duration_s
    # seconds of that file are processed so you can inspect a short slice without
    # waiting for a full capture to run through the pipeline.
    "debug_plots": False,
    "debug_file_num": 0,       # Index of the file to inspect in debug mode
    "debug_duration_s": None,  # How many seconds of IQ to process (None = full file)
}


class SpectReProcessor:
    """
    Handles batch processing of IQ capture files for SpectRe.
    
    For each file, the pipeline:
      1. Loads IQ samples and syncs compass headings
      2. Shifts the beacon signal to baseband
      3. Filters and decimates
      4. Runs the matched filter (LoRa detector)
      5. Extracts the densest LOB region and associated heading
      6. Accumulates bearing estimates for localization
    """

    def __init__(self, config: dict):
        self.config = config

    def process(self) -> list:
        """
        Runs the processing pipeline.

        In normal mode (debug_plots=False): iterates over all files in num_files
        and accumulates LOB estimates for localization.

        In debug mode (debug_plots=True): processes only debug_file_num, truncated
        to debug_duration_s seconds of IQ, then shows all stage plots and returns.
        Batch localization is skipped — debug mode is for pipeline inspection only.

        Returns:
            List of bearing estimate dicts, one per detected LOB.
        """
        plt.close('all')
        all_estimates = []

        if self.config["debug_plots"]:
            # --- Debug mode: single file, time-limited ---
            file_num = self.config["debug_file_num"]
            print(f"\n[DEBUG] Single-file mode: file {file_num} | "
                  f"duration={self.config['debug_duration_s']}s")
            estimates = self._process_file(file_num)
            all_estimates.extend(estimates)
        else:
            # --- Batch mode: all files ---
            for file_num in range(self.config["num_files"]):
                estimates = self._process_file(file_num)
                all_estimates.extend(estimates)

        print(f"\n[*] Batch processing complete. Total estimates saved: {len(all_estimates)}")
        return all_estimates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_file(self, file_num: int) -> list:
        """
        Loads a single capture file, splits it into test segments,
        and runs the DSP pipeline on each segment.

        Args:
            file_num: Index of the capture file to process.

        Returns:
            List of bearing estimate dicts extracted from this file.
        """
        estimates = []

        meta_filename = f"{self.config['file_prefix']}{file_num}.json"
        meta_path = os.path.join(self.config["folder_path"], meta_filename)

        if not os.path.exists(meta_path):
            print(f"[!] Skipping: {meta_path} (File not found)")
            return estimates

        print(f"\n[*] Processing File: {meta_path}")

        # --- Load metadata and resolve file paths ---
        meta = extract_capture_metadata(meta_path)
        loader = IQDataManager(meta)
        viz = IQVisualizer(debug_mode=self.config["debug_plots"])

        iq_filename = os.path.join(self.config["folder_path"], meta.get('IQ_file', ''))
        compass_csv = os.path.join(self.config["folder_path"], meta.get('heading_file', ''))

        # --- Extract hardware parameters from metadata ---
        lat = meta.get('lat')
        lon = meta.get('lon')
        fs_actual = meta.get('rate')
        fc_actual = meta.get('freq')
        compass_offset = meta.get('log_delta_ns')  # Hardware timing offset between IQ and compass

        if not os.path.exists(iq_filename):
            print(f"[!] ERROR: Could not find IQ file: {iq_filename}")
            return estimates

        # --- Load full IQ and sync compass headings ---
        iq_full = loader.load_iq(iq_filename, format_type=self.config["iq_format"])

        # --- Debug mode: truncate to debug_duration_s seconds ---
        # This lets you inspect a short slice of a long capture without waiting
        # for the full file to run through the pipeline.
        if self.config["debug_plots"] and self.config.get("debug_duration_s") is not None:
            max_samples = int(self.config["debug_duration_s"] * fs_actual)
            if max_samples < len(iq_full):
                print(f"  [DEBUG] Truncating to {self.config['debug_duration_s']}s "
                      f"({max_samples} / {len(iq_full)} samples)")
                iq_full = iq_full[:max_samples]

        if os.path.exists(compass_csv):
            # Sync compass timestamps to IQ sample timeline using hardware delay offset
            headings_full = loader.sync_sensors(compass_csv, len(iq_full), delay_offset=compass_offset * 1e-9)
        else:
            # Fall back to final recorded heading if no compass CSV is available
            headings_full = np.full(len(iq_full), meta.get('heading_final', 0))

        # --- Split capture into N test segments (one per rotation) ---
        samples_per_test = len(iq_full) // self.config["tests_per_file"]

        for test_idx in range(self.config["tests_per_file"]):
            print(f"  [>] Test {test_idx + 1}/{self.config['tests_per_file']}")

            start_idx = test_idx * samples_per_test
            end_idx = (test_idx + 1) * samples_per_test

            iq = iq_full[start_idx:end_idx]
            headings = headings_full[start_idx:end_idx]

            weight = calculate_iq_weight(iq)

            estimate = self._run_dsp_pipeline(
                iq, headings, fs_actual, fc_actual,
                meta_filename, file_num, test_idx,
                lat, lon, viz
            )

            if estimate:
                estimates.append(estimate)

        return estimates

    def _run_dsp_pipeline(
        self,
        iq: np.ndarray,
        headings: np.ndarray,
        fs_actual: float,
        fc_actual: float,
        meta_filename: str,
        file_num: int,
        test_idx: int,
        lat: float,
        lon: float,
        viz: IQVisualizer
    ) -> dict | None:
        """
        Runs the DSP pipeline on a single test segment:
          baseband shift → filter → decimate → matched filter → LOB extraction.

        Args:
            iq:            Complex IQ samples for this segment.
            headings:      Compass headings synced to the IQ timeline.
            fs_actual:     Actual SDR sample rate (Hz).
            fc_actual:     Actual SDR center frequency (Hz).
            meta_filename: Source metadata filename (for logging).
            file_num:      File index (for test ID tagging).
            test_idx:      Segment index within the file.
            lat:           Receiver latitude from metadata.
            lon:           Receiver longitude from metadata.
            viz:           IQVisualizer instance for debug plots.

        Returns:
            Bearing estimate dict if a LOB was detected, else None.
        """

        # --- Step 1: Baseband shift → filter → decimate ---
        # debug_plots=True will plot spectrum, time series, and constellation
        # at every stage: raw → post-shift → post-filter → post-decimate
        sdc = SignalDownConverter(fs=fs_actual)
        freq_offset = self.config["target_freq"] - fc_actual
        iq_bb, fs_bb = sdc.process_pipeline(
            iq,
            offset_hz=freq_offset,
            bw_target=self.config["lora_bw"],
            decimation_factor=self.config["decimation"],
            debug=self.config["debug_plots"],
            fc_actual=fc_actual
        )

        # Decimate compass headings to match the new sample rate
        headings_bb = headings[::self.config["decimation"]]

        # --- Step 2: Matched filter detection ---
        # Correlate received signal against reference FMCW chirp template
        detector = LoRaDetector(
            sf=self.config["lora_sf"],
            bw=self.config["lora_bw"],
            fs=fs_bb,
            window_size=self.config["window_size"]
        )
        corr, regions = detector.detect(
            iq_bb,
            threshold=self.config["threshold"],
            normalize=True,
            debug=self.config["debug_plots"]   # plots stages 1–6 when enabled
        )

        # --- Step 3: LOB extraction ---
        # Find the densest cluster of chirp detections (peak of the spin arc)
        lob_data = detector.get_robust_lob_region(
            regions,
            debug=self.config["debug_plots"]   # plots cluster scoring + timeline when enabled
        )

        if lob_data:
            offset = self.config.get("bearing_offset_deg", 0.0)

            # Apply calibration bearing offset to correct for antenna/compass misalignment
            h_start = (headings_bb[int(lob_data['start'])] + offset) % 360
            h_stop  = (headings_bb[int(lob_data['stop'])]  + offset) % 360

            raw_center = headings_bb[int((lob_data['start'] + lob_data['stop']) / 2)]
            h_peak = (raw_center + offset) % 360

            estimate = {
                "source":          meta_filename,
                "test_id":         f"{file_num}_{test_idx}",
                "lat":             lat,
                "lon":             lon,
                "heading_start":   h_start,
                "heading_end":     h_stop,
                "heading_center":  h_peak,
                "max_corr":        np.max(corr),
                "weight":          1
            }

            print(f"    [+] LOB Detected: {h_start:.1f}° to {h_stop:.1f}° (Peak: {h_peak:.1f}°)")

            # --- Debug plots ---
            # Signal_Downconverter and LoRaDetector have already queued their figures
            # (DDC stages 0–3, MF stages 1–6, LOB clustering) without blocking.
            # Here we add the IQ Visualizer plots that sit outside those classes,
            # then call plt.show() once so every figure for this segment renders together.
            if self.config["debug_plots"]:
                # Spectrogram of raw IQ — complements DDC Stage 0 spectrum
                view_spectrogram(iq, fs_actual)

                # LOB detection overlay on correlation output with heading axis
                if len(regions) > 0:
                    viz.plot_detections(corr, regions, fs_bb, headings_bb,
                                        title=f"LOB Results - {meta_filename} [{test_idx}]")

                # Signal quality summary (SNR, power, symbol alignment)
                plot_signal_quality(iq, fs_actual, detector.num_samples_per_symbol)

                # --- Uncomment for additional IQ diagnostics ---
                # viz.plot_constellation(iq)
                # viz.plot_histo(iq)
                # viz.plot_time(iq, fs_actual, 0, 1)

                # Single show call — blocks here until all figures are closed
                plt.show()

            return estimate

        return None