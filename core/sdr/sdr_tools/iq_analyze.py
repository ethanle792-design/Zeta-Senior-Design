#!/usr/bin/env python3
"""
iq_analyze.py — offline FFT/PSD for .cf32 captures.

Example:
  python3 iq_analyze.py --file capture_*.cf32 --rate 5e6 --freq 915e6
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Analyze CF32 IQ capture (FFT/PSD).")
    p.add_argument("--file", required=True, help="Path to .cf32 file")
    p.add_argument("--rate", type=float, required=True, help="Sample rate in S/s (e.g. 5e6)")
    p.add_argument("--freq", type=float, required=True, help="Center frequency in Hz (e.g. 915e6)")
    p.add_argument("--nfft", type=int, default=4096, help="FFT size (default: 4096)")
    p.add_argument("--skip", type=int, default=0, help="Skip initial complex samples (default: 0)")
    p.add_argument("--count", type=int, default=0, help="Analyze only this many samples (0=all)")
    return p.parse_args()


def main():
    args = parse_args()
    path = args.file
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = np.fromfile(path, dtype=np.complex64)
    if args.skip > 0:
        raw = raw[args.skip:]
    if args.count and args.count > 0:
        raw = raw[:args.count]

    if raw.size < args.nfft:
        raise ValueError(f"Not enough samples ({raw.size}) for nfft={args.nfft}")

    # Replace the FFT section with this

    nfft = args.nfft
    rate = args.rate

    # Trim to integer number of FFT blocks
    num_blocks = len(raw) // nfft
    if num_blocks == 0:
        raise ValueError("Not enough samples for one FFT block.")

    raw = raw[:num_blocks * nfft]
    blocks = raw.reshape((num_blocks, nfft))

    window = np.hanning(nfft).astype(np.float32)

    psd_accum = np.zeros(nfft, dtype=np.float64)

    for block in blocks:
        xw = block * window
        X = np.fft.fftshift(np.fft.fft(xw))
        psd = np.abs(X) ** 2
        psd_accum += psd

    psd_avg = psd_accum / num_blocks
    psd_db = 10 * np.log10(psd_avg + 1e-15)

    # Frequency axis
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate))
    f_hz = args.freq + f

    plt.figure()
    plt.plot(f_hz / 1e6, psd_db)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB, relative)")
    plt.title("Averaged PSD")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
