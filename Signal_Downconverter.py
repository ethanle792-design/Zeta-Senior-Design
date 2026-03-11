import numpy as np
from scipy.signal import firwin, filtfilt

class SignalDownConverter:
    def __init__(self, fs):
        """
        Args:
            fs (float): The initial sample rate of the input IQ data.
        """
        self.fs = fs

    def shift_to_baseband(self, iq_samples, offset_hz):
        """
        Shifts a target frequency offset to 0 Hz (Complex Mixing).
        
        Args:
            iq_samples (np.array): Complex IQ data.
            offset_hz (float): The frequency distance from center (Target - Center).
        """
        t = np.arange(len(iq_samples)) / self.fs
        # Multiplying by -offset_hz "un-spins" the target back to 0 Hz
        mixer = np.exp(-1j * 2.0 * np.pi * offset_hz * t)
        return iq_samples * mixer

    def lowpass_filter(self, iq, f_cutoff, numtaps=101, window='hamming'):
        """
        Applies a zero-phase FIR lowpass filter. 
        Crucial for removing alias components before decimation.
        """
        nyq = self.fs / 2
        f_cutoff_norm = f_cutoff / nyq
        
        # Design filter taps
        taps = firwin(numtaps, cutoff=f_cutoff_norm, window=window)

        # filtfilt ensures zero time-delay (essential for timing/geolocation)
        return filtfilt(taps, 1.0, iq)

    def decimate(self, iq, factor=10):
        """
        Reduces the sample rate by keeping every M-th sample.
        
        Returns:
            decimated_iq (np.array): The downsampled signal.
            new_fs (float): The updated sample rate.
        """
        decimated_iq = iq[::factor]
        new_fs = self.fs / factor
        
        # Update internal state so subsequent operations use the new rate
        self.fs = new_fs
        
        return decimated_iq, new_fs

    def process_pipeline(self, iq, offset_hz, bw_target, decimation_factor):
        """
        A helper to run a full DDC chain: Shift -> Filter -> Decimate.
        """
        # 1. Shift target to 0 Hz
        iq = self.shift_to_baseband(iq, offset_hz)
        
        # 2. Filter out everything except the target bandwidth
        # We filter at half the target bandwidth because it's complex IQ (Double Sided)
        iq = self.lowpass_filter(iq, f_cutoff=bw_target/2)
        
        # 3. Reduce sample rate
        iq, new_fs = self.decimate(iq, factor=decimation_factor)
        
        return iq, new_fs