import numpy as np
import scipy.signal as signal

def generate_lora_chirp(sf, bw, fs):
    """
    Generates a standard LoRa 'Upchirp' in the time domain.
    """
    # Calculate duration of one symbol
    duration = (2**sf) / bw
    
    # Create time array
    num_samples = int(duration * fs)
    t = np.arange(num_samples) / fs
    
    # Linear Chirp Math: exp(j * pi * (bw/T) * t^2)
    # This sweeps from 0 to BW. To center it (-BW/2 to +BW/2), 
    # For a standard LoRa upchirp centered at baseband:
    k = bw / duration
    chirp = np.exp(1j * 2 * np.pi * ((-bw/2) * t + 0.5 * k * t**2))
    
    return chirp

def matched_filter(chirp, rx_signal, normalize=True):
    """
    Fast Matched Filter using FFT convolution.
    
    Args:
        chirp: LoRa chirp template (Complex IQ)
        rx_signal: Received signal (Complex IQ)
    """
    # 1. Create the Matched Filter Template
    # Conjugate and Time-Reverse the chirp
    h = np.conj(chirp[::-1])

    # 2. Fast Convolution (FFT Method)
    # 'same' mode automatically centers the result
    # 'full' is safer if you want total control over indices, 
    # but 'same' is usually fine for generic peak finding.
    corr_full = signal.fftconvolve(rx_signal, h, mode='full')
    
    # 3. Trim (Optional: Aligning 'full' output to input)
    # Convolution makes the array longer (N + M - 1). 
    # This keeps the valid region consistent.
    N = len(chirp)
    # We slice to keep the output size roughly equal to rx_signal
    # The peak will appear exactly when the chirp ends in the rx stream
    corr = corr_full[N-1 : N-1 + len(rx_signal)]

    # 4. Return RAW Magnitude (for DF) or Normalized (for visualization)
    if normalize:
        max_val = np.max(np.abs(corr)) + 1e-12
        corr /= max_val
        
    return corr

def extract_corr_regions(corr, threshold):
    """
    Find contiguous regions in correlation above a threshold.

    Args:
        corr: 1D array (normalized correlation)
        threshold: float, 0..1

    Returns:
        regions: list of tuples [(start_idx, stop_idx), ...]
    """
    corr = np.asarray(corr)
    above = corr >= threshold

    # find rising/falling edges
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1  # rising edge
    stops  = np.where(edges == -1)[0] + 1 # falling edge

    # handle edge cases
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        stops = np.append(stops, len(corr))

    # pair them
    regions = list(zip(starts, stops))
    return regions


