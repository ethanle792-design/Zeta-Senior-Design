import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt


class SignalDownConverter:
    def __init__(self, fs: float):
        """
        Args:
            fs: Initial sample rate of the input IQ data (Hz).
        """
        self.fs = fs

    # ==========================================
    # --- DSP STAGES ---
    # ==========================================

    def shift_to_baseband(self, iq_samples: np.ndarray, offset_hz: float) -> np.ndarray:
        """
        Shifts a target frequency offset to 0 Hz via complex mixing.

        Multiplying by exp(-j2π·offset·t) rotates the signal in the complex
        plane at the offset rate, effectively "un-spinning" the target to DC.

        Args:
            iq_samples: Complex IQ data.
            offset_hz:  Frequency distance from center (target_freq - center_freq).

        Returns:
            Baseband-shifted complex IQ array.
        """
        t = np.arange(len(iq_samples)) / self.fs
        mixer = np.exp(-1j * 2.0 * np.pi * offset_hz * t)
        return iq_samples * mixer

    def lowpass_filter(self, iq: np.ndarray, f_cutoff: float,
                       numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        Applies a zero-phase FIR lowpass filter.

        filtfilt eliminates group delay — critical for heading timestamp
        alignment in the LOB extraction stage.

        Args:
            iq:       Complex IQ data.
            f_cutoff: Cutoff frequency (Hz), relative to current self.fs.
            numtaps:  FIR filter length.
            window:   Window function for firwin.

        Returns:
            Filtered complex IQ array.
        """
        nyq = self.fs / 2
        f_cutoff_norm = f_cutoff / nyq
        taps = firwin(numtaps, cutoff=f_cutoff_norm, window=window)
        return filtfilt(taps, 1.0, iq)

    def decimate(self, iq: np.ndarray, factor: int = 10) -> tuple[np.ndarray, float]:
        """
        Decimates the IQ stream by keeping every Nth sample.

        Must always be preceded by a lowpass filter to prevent aliasing.
        self.fs is not mutated — the new sample rate is returned alongside
        the decimated array so the caller controls state.

        Args:
            iq:     Filtered complex IQ data.
            factor: Decimation factor.

        Returns:
            (decimated_iq, new_fs) tuple.
        """
        if factor <= 1:
            return iq, self.fs
        return iq[::factor], self.fs / factor

    # ==========================================
    # --- PIPELINE ---
    # ==========================================

    def process_pipeline(self, iq: np.ndarray, offset_hz: float,
                         bw_target: float, decimation_factor: int,
                         debug: bool = False,
                         fc_actual: float = None) -> tuple[np.ndarray, float]:
        """
        Full DDC chain: Shift → Filter → Decimate.

        Args:
            iq:                Raw complex IQ samples.
            offset_hz:         Frequency offset to shift to baseband (Hz).
            bw_target:         Target signal bandwidth (Hz).
            decimation_factor: Sample rate reduction factor.
            debug:             If True, plots spectrum and IQ at every stage.
            fc_actual:         SDR center frequency (Hz). Used to label
                               the raw spectrum x-axis. Optional.

        Returns:
            (baseband_iq, new_sample_rate) tuple.
        """
        print(f"  [DDC] fs={self.fs/1e3:.1f} kHz | offset={offset_hz/1e3:.1f} kHz | "
              f"bw={bw_target/1e3:.1f} kHz | decimate={decimation_factor}x")

        if debug:
            self._plot_stage(iq, self.fs,
                             center_hz=fc_actual,
                             title="Stage 0: Raw IQ (pre-shift)",
                             note=f"Center: {fc_actual/1e6:.3f} MHz" if fc_actual else "")

        # --- Stage 1: Baseband shift ---
        iq = self.shift_to_baseband(iq, offset_hz)

        if debug:
            self._plot_stage(iq, self.fs,
                             center_hz=0,
                             title="Stage 1: After Baseband Shift",
                             note=f"Shifted by {offset_hz/1e3:.1f} kHz → target now at DC")

        # --- Stage 2: Lowpass filter ---
        # Cutoff at half the target BW — IQ is double-sided so BW/2 isolates the signal
        iq = self.lowpass_filter(iq, f_cutoff=bw_target / 2)

        if debug:
            self._plot_stage(iq, self.fs,
                             center_hz=0,
                             title="Stage 2: After Lowpass Filter",
                             note=f"Cutoff: ±{bw_target/2/1e3:.1f} kHz  |  taps=101  |  zero-phase (filtfilt)")

        # --- Stage 3: Decimation ---
        iq, new_fs = self.decimate(iq, factor=decimation_factor)

        if debug:
            self._plot_stage(iq, new_fs,
                             center_hz=0,
                             title="Stage 3: After Decimation",
                             note=f"Rate: {self.fs/1e3:.1f} kHz → {new_fs/1e3:.1f} kHz  "
                                  f"(÷{decimation_factor})  |  signal BW={bw_target/1e3:.1f} kHz")
            # plt.show() intentionally omitted — caller (SpectReProcessor) owns the show call
            # so all DDC and detector figures render together before blocking

        return iq, new_fs

    # ==========================================
    # --- DEBUG PLOTTING ---
    # ==========================================

    def _plot_stage(self, iq: np.ndarray, fs: float,
                    center_hz: float = None,
                    title: str = "",
                    note: str = "") -> None:
        """
        Plots three panels for a single pipeline stage:
          - Power spectrum  (FFT, x-axis in actual Hz)
          - IQ time series  (I and Q overlaid)
          - Spectrogram     (scipy STFT on complex IQ, axes in actual Hz)

        center_hz controls axis labelling:
          Stage 0  → pass fc_actual (e.g. 915.5e6): both spectrum and spectrogram
                     x/y-axes read in absolute RF MHz.
          Stages 1–3 → pass 0: axes read in kHz offset from DC.

        Args:
            iq:        Complex IQ samples at this stage.
            fs:        Sample rate at this stage (Hz).
            center_hz: Frequency at DC (Hz). fc_actual for raw; 0 for baseband stages.
            title:     Figure title (stage name).
            note:      Subtitle with parameter details.
        """
        from scipy.signal import stft as scipy_stft

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(title, fontsize=12, fontweight='bold')
        if note:
            fig.text(0.5, 0.92, note, ha='center', fontsize=9, color='gray')

        n = len(iq)
        c = center_hz if center_hz is not None else 0.0
        use_mhz = c > 1e6  # RF stage → MHz labels; baseband → kHz labels
        x_unit = "MHz" if use_mhz else "kHz"

        def to_display(f_hz):
            """Convert an absolute-Hz value to the display unit (MHz or kHz)."""
            return f_hz / 1e6 if use_mhz else f_hz / 1e3

        # ---- Panel 1: FFT power spectrum ------------------------------------
        # Full FFT over the entire IQ array using a Blackman window to reduce
        # spectral leakage. fftshift centres DC so negative frequencies are on
        # the left, matching SDR convention.
        ax = axes[0]
        fft_size = len(iq)
        win = np.blackman(fft_size)
        spectrum = np.fft.fftshift(np.fft.fft(iq * win, fft_size))
        mag_db = 20 * np.log10(np.abs(spectrum) / fft_size + 1e-12)

        # fftfreq returns bin offsets from DC in Hz; add c for absolute frequency
        bin_offsets_hz = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / fs))
        freqs_display = to_display(bin_offsets_hz + c)

        ax.plot(freqs_display, mag_db, linewidth=0.7, color='steelblue')
        ax.set_xlabel(f"Frequency ({x_unit})")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"FFT Spectrum ({fft_size} samples)")
        ax.grid(True, alpha=0.3)

        # ---- Panel 2: IQ time series ----------------------------------------
        ax = axes[1]
        t_us = np.arange(n) / fs * 1e6
        ax.plot(t_us, np.real(iq), linewidth=0.6, label='I', color='steelblue')
        ax.plot(t_us, np.imag(iq), linewidth=0.6, label='Q', color='coral', alpha=0.8)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"IQ Time Series ({n} samples)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Panel 3: Spectrogram -------------------------------------------
        # scipy.signal.stft handles complex IQ correctly — it does not discard
        # the Q channel the way matplotlib's specgram does with complex input.
        # This gives a two-sided spectrum centred at DC with negative frequencies
        # on the left, matching what the SDR actually captured.
        # nperseg=256 gives ~0.4ms time resolution at 600kHz; increase for better
        # frequency resolution at the cost of time resolution.
        ax = axes[2]
        nperseg = 256

        f_stft, t_stft, Zxx = scipy_stft(
            iq,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,   # complex IQ → two-sided spectrum
            boundary=None
        )

        # fftshift centres DC; add c to get absolute frequency, then scale
        f_stft_shifted = np.fft.fftshift(f_stft)          # DC-centred bin offsets (Hz)
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)        # match row order
        f_display = to_display(f_stft_shifted + c)         # absolute, display unit

        power_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)

        im = ax.pcolormesh(t_stft, f_display, power_db,
                           shading='auto', cmap='viridis')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Frequency ({x_unit})")
        ax.set_title(f"Spectrogram ({n} samples)")
        fig.colorbar(im, ax=ax, label="Power (dB)")

        plt.tight_layout(rect=[0, 0, 1, 0.90])