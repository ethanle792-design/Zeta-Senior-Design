#!/usr/bin/env python3
"""
iq_logger.py — robust SoapySDR I/Q capture to disk with JSON metadata sidecar.

Outputs:
  - <base>.cf32  : raw complex64 samples (IQ interleaved as np.complex64)
  - <base>.json  : capture metadata (freq/rate/gain/device info, stats, placeholders for GPS/heading)

Example:
  python3 iq_logger.py --freq 915e6 --rate 5e6 --gain 40 --duration 3 --out ./captures

Replay/analyze:
  python3 iq_analyze.py --file ./captures/capture_YYYYMMDD_HHMMSSZ.cf32 --rate 5e6 --freq 915e6
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # type: ignore
except ImportError:
    print("ERROR: SoapySDR python bindings not found. Try: pip3 install SoapySDR (or apt packages).")
    sys.exit(1)


STOP = False


def _sigint_handler(signum, frame):
    global STOP
    STOP = True


def utc_ts_compact() -> str:
    # e.g., 20260225_183012Z
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="SoapySDR I/Q logger (CF32 + JSON metadata).")
    p.add_argument("--device", default="driver=bladerf", help="SoapySDR device args string (default: driver=bladerf)")
    p.add_argument("--channel", type=int, default=0, help="RX channel index (default: 0)")
    p.add_argument("--freq", type=float, required=True, help="Center frequency in Hz (e.g. 915e6)")
    p.add_argument("--rate", type=float, required=True, help="Sample rate in S/s (e.g. 5e6)")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB (if supported). If omitted, leaves default.")
    p.add_argument("--bandwidth", type=float, default=None, help="Optional bandwidth in Hz (if supported).")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", type=float, help="Capture duration in seconds (e.g. 3)")
    group.add_argument("--samples", type=int, help="Capture exactly N complex samples")

    p.add_argument("--out", default="./captures", help="Output directory (default: ./captures)")
    p.add_argument("--name", default=None, help="Base name (default: capture_<UTCtimestamp>)")

    p.add_argument("--block", type=int, default=262144,
                   help="Read block size in complex samples (default: 262144). Larger = fewer syscalls.")
    p.add_argument("--timeout-us", type=int, default=250000,
                   help="Stream read timeout in microseconds (default: 250000)")
    p.add_argument("--settle-ms", type=int, default=150,
                   help="Time to let tuner settle before streaming (default: 150ms)")

    # Placeholders for later integration (don’t wire sensors yet, just reserve fields)
    p.add_argument("--tag", default=None, help="Optional tag for metadata (e.g. 'scan1_north_lot')")
    return p.parse_args()


def main():
    global STOP
    signal.signal(signal.SIGINT, _sigint_handler)

    args = parse_args()
    safe_mkdir(args.out)

    base = args.name or f"capture_{utc_ts_compact()}"
    cf32_path = os.path.join(args.out, base + ".cf32")
    json_path = os.path.join(args.out, base + ".json")

    # Connect SDR
    print(f"[+] Creating SDR: {args.device}")
    sdr = SoapySDR.Device(args.device)

    ch = args.channel

    # Configure
    print(f"[+] Config RX ch{ch}: freq={args.freq} Hz rate={args.rate} S/s gain={args.gain} bw={args.bandwidth}")
    sdr.setSampleRate(SOAPY_SDR_RX, ch, args.rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, args.freq)

    if args.bandwidth is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, ch, args.bandwidth)
        except Exception as e:
            print(f"[!] Bandwidth set failed (ignored): {e}")

    if args.gain is not None:
        try:
            # Some drivers prefer setGain, some use element gains; start simple.
            sdr.setGain(SOAPY_SDR_RX, ch, args.gain)
        except Exception as e:
            print(f"[!] Gain set failed (ignored): {e}")

    # Read back what we actually got
    actual = {
        "freq_hz": float(sdr.getFrequency(SOAPY_SDR_RX, ch)),
        "rate_sps": float(sdr.getSampleRate(SOAPY_SDR_RX, ch)),
    }
    try:
        actual["bandwidth_hz"] = float(sdr.getBandwidth(SOAPY_SDR_RX, ch))
    except Exception:
        actual["bandwidth_hz"] = None

    try:
        actual["gain_db"] = float(sdr.getGain(SOAPY_SDR_RX, ch))
    except Exception:
        actual["gain_db"] = args.gain

    # Stream setup
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])

    # Allow PLL settling / AGC settling (if any)
    if args.settle_ms > 0:
        time.sleep(args.settle_ms / 1000.0)

    sdr.activateStream(rx_stream)

    # Capture planning
    if args.samples is not None:
        target_samps = int(args.samples)
        target_mode = "samples"
    else:
        target_samps = int(round(args.duration * actual["rate_sps"]))
        target_mode = "duration"

    block = int(args.block)
    buf = np.empty(block, dtype=np.complex64)

    # Stats
    t_start = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()
    total_written = 0
    timeouts = 0
    overruns = 0
    read_errors = 0

    print(f"[+] Writing to: {cf32_path}")
    print(f"[+] Target: {target_samps} complex samples ({target_mode})")
    print("[+] Capturing... Ctrl+C to stop early.")

    with open(cf32_path, "wb", buffering=0) as f:
        while not STOP and total_written < target_samps:
            # Figure how many we still need for the final partial block
            need = min(block, target_samps - total_written)

            sr = sdr.readStream(rx_stream, [buf], need, timeoutUs=args.timeout_us)

            if sr.ret > 0:
                # Write only the amount read
                f.write(buf[:sr.ret].tobytes())
                total_written += sr.ret
                continue

            # sr.ret <= 0 indicates issues. Interpret common ones.
            if sr.ret == 0:
                timeouts += 1
                continue

            # Negative return codes vary by driver; flags may include overflow/overrun info.
            read_errors += 1
            # Some Soapy drivers set sr.flags to indicate overflow/overrun
            try:
                if sr.flags is not None and (sr.flags & SoapySDR.SOAPY_SDR_END_ABRUPT) != 0:
                    overruns += 1
            except Exception:
                pass

            # Don’t spin too hard if errors persist
            time.sleep(0.001)

    # Stop stream
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    t_end = time.time()
    end_utc = datetime.now(timezone.utc).isoformat()

    # Metadata sidecar (GPS/heading placeholders included for later)
    meta = {
        "base": base,
        "file_cf32": os.path.basename(cf32_path),
        "format": "cf32",
        "dtype": "complex64",
        "device_args": args.device,
        "rx_channel": ch,
        "requested": {
            "freq_hz": args.freq,
            "rate_sps": args.rate,
            "gain_db": args.gain,
            "bandwidth_hz": args.bandwidth,
            "duration_s": args.duration,
            "samples": args.samples,
            "block_samples": block,
            "timeout_us": args.timeout_us,
        },
        "actual": actual,
        "timing": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "elapsed_s": t_end - t_start,
        },
        "stats": {
            "samples_written": total_written,
            "complex_samples_target": target_samps,
            "timeouts": timeouts,
            "overruns": overruns,
            "read_errors": read_errors,
        },
        "tag": args.tag,
        "sensors": {
            # Reserved for later integration
            "gps": {"lat": None, "lon": None, "alt_m": None, "fix": None},
            "heading_deg": None,
        },
        "notes": [
            "Raw IQ is complex64 interleaved (I+jQ) at the given sample rate.",
            "Use iq_analyze.py to replay/FFT without needing the SDR attached.",
        ],
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    print("\n[+] Done.")
    print(f"    Samples written: {total_written}")
    print(f"    Elapsed: {t_end - t_start:.3f} s")
    print(f"    Metadata: {json_path}")
    if STOP:
        print("    Stopped early by user (Ctrl+C).")


if __name__ == "__main__":
    main()
