#!/usr/bin/env python3
"""
sdr_log.py - Log LimeSDR signal strength vs frequency using SoapySDR

Modes:
  1) Fixed frequency:
      - Stay on one center frequency
      - Log average power every interval
      Example:
        python3 sdr_log.py --mode fixed --freq 915e6 --rate 1e6 --gain 20

  2) Sweep:
      - Sweep a frequency range [start, stop] with a given step
      - Log one power measurement per frequency
      Example:
        python3 sdr_log.py --mode sweep --f-start 900e6 --f-stop 930e6 --f-step 1e6

Logs to CSV (default: sdr_log.csv) with columns:
  timestamp_iso, freq_hz, power_db, gps_lat, gps_lon, gps_alt

GPS fields are placeholders for later Matek integration.
"""

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


# ---------------------------
# GPS stub (for later)
# ---------------------------
def get_gps_fix():
    """
    Placeholder for Matek GNSS integration.
    Return (lat, lon, alt) or (None, None, None) if unavailable.
    """
    # TODO: wire this up to the Matek module later
    return None, None, None


def parse_args():
    p = argparse.ArgumentParser(description="LimeSDR signal strength logger")

    # General SDR settings
    p.add_argument("--mode", choices=["fixed", "sweep"], default="fixed",
                   help="Operation mode: fixed or sweep (default: fixed)")
    p.add_argument("--rate", type=float, default=1e6,
                   help="Sample rate in samples/sec (default: 1e6)")
    p.add_argument("--gain", type=float, default=20.0,
                   help="RX gain in dB (default: 20)")
    p.add_argument("--antenna", type=str, default="LNAL",
                   help="Antenna name (LNAL/LNAH/LNAW/etc, default: LNAL)")
    p.add_argument("--nsamps", type=int, default=4096,
                   help="Samples per measurement (default: 4096)")

    # Fixed frequency mode
    p.add_argument("--freq", type=float, default=915e6,
                   help="Center frequency in Hz for fixed mode (default: 915e6)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between measurements in fixed mode (default: 1.0)")

    # Sweep mode
    p.add_argument("--f-start", type=float, default=900e6,
                   help="Sweep start frequency in Hz (default: 900e6)")
    p.add_argument("--f-stop", type=float, default=930e6,
                   help="Sweep stop frequency in Hz (default: 930e6)")
    p.add_argument("--f-step", type=float, default=1e6,
                   help="Sweep step in Hz (default: 1e6)")
    p.add_argument("--sweep-delay", type=float, default=0.05,
                   help="Seconds to wait after tuning before measuring (default: 0.05)")

    # Logging
    p.add_argument("--out", type=str, default="sdr_log.csv",
                   help="Output CSV file (default: sdr_log.csv)")
    p.add_argument("--append", action="store_true",
                   help="Append to existing CSV instead of overwriting")

    return p.parse_args()


def open_sdr(args):
    print("[SDR] Enumerating devices...")
    devs = SoapySDR.Device.enumerate()
    print(f"[SDR] Devices found: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"  [{i}] {d}")
    if not devs:
        raise RuntimeError("No SoapySDR devices found. Check SoapySDRUtil --find.")

    print("[SDR] Opening device 0...")
    sdr = SoapySDR.Device(devs[0])

    chan = 0
    print("[SDR] Configuring RX...")
    sdr.setSampleRate(SOAPY_SDR_RX, chan, args.rate)
    # freq will be set per-measurement (especially in sweep mode)
    sdr.setGain(SOAPY_SDR_RX, chan, args.gain)

    try:
        sdr.setAntenna(SOAPY_SDR_RX, chan, args.antenna)
        print(f"[SDR] Using antenna: {args.antenna}")
    except Exception as e:
        print(f"[SDR] Warning: could not set antenna to {args.antenna}: {e}")

    try:
        sdr.setBandwidth(SOAPY_SDR_RX, chan, args.rate)
    except Exception:
        pass

    # Setup stream
    print("[SDR] Setting up RX stream...")
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])
    sdr.activateStream(stream)
    time.sleep(0.1)

    return sdr, stream, chan


def close_sdr(sdr, stream):
    try:
        sdr.deactivateStream(stream)
    except Exception:
        pass
    try:
        sdr.closeStream(stream)
    except Exception:
        pass


def measure_power(sdr, stream, chan, freq_hz, nsamps, settle_time=0.02):
    """
    Tune to freq_hz, grab nsamps samples, and return average power in dBFS-ish.
    """
    # Tune
    sdr.setFrequency(SOAPY_SDR_RX, chan, freq_hz)
    time.sleep(settle_time)

    buf = np.empty(nsamps, dtype=np.complex64)

    # Read a block
    sr = sdr.readStream(stream, [buf], nsamps, timeoutUs=int(1e6))
    if sr.ret <= 0:
        # read failed, return sentinel
        return None

    samples = buf[:sr.ret]

    # Compute power: mean(|x|^2), convert to dB
    power_lin = np.mean(np.abs(samples) ** 2)
    power_db = 10 * np.log10(power_lin + 1e-20)  # avoid log(0)

    return power_db


def setup_csv(path, append=False):
    mode = "a" if append else "w"
    f = open(path, mode, newline="")
    writer = csv.writer(f)

    if not append or f.tell() == 0:
        # Write header
        writer.writerow(["timestamp_iso", "freq_hz", "power_db",
                         "gps_lat", "gps_lon", "gps_alt"])

    return f, writer


def run_fixed_mode(args, sdr, stream, chan, writer):
    print("[MODE] Fixed frequency logging")
    print(f"       Center freq: {args.freq/1e6:.3f} MHz")
    print(f"       Interval:    {args.interval:.2f} s")

    try:
        while True:
            t = datetime.utcnow().isoformat()
            power_db = measure_power(sdr, stream, chan, args.freq,
                                     args.nsamps, settle_time=0.0)
            gps_lat, gps_lon, gps_alt = get_gps_fix()

            if power_db is None:
                print(f"{t}  freq={args.freq/1e6:.3f} MHz  power=READ_FAIL")
            else:
                print(f"{t}  freq={args.freq/1e6:.3f} MHz  power={power_db:.2f} dB")

            writer.writerow([t, int(args.freq), power_db,
                             gps_lat, gps_lon, gps_alt])

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[MODE] Fixed mode stopped by user.")


def run_sweep_mode(args, sdr, stream, chan, writer):
    print("[MODE] Sweep logging")
    print(f"       Start: {args.f-start/1e6:.3f} MHz")  # this line will error: fixed below
