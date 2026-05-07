#!/usr/bin/env python3
"""
iq_logger_compass.py — SoapySDR I/Q capture (CF32) + compass logging (CSV) + JSON metadata.

Outputs:
  - <base>.cf32          : raw complex64 samples
  - <base>.json          : capture metadata (includes compass csv filename + last heading)
  - <base>_heading.csv   : compass samples (timestamped relative to capture start)

Example:
  python3 iq_logger_compass.py --freq 915e6 --rate 5e6 --gain 40 --duration 5 --out ./captures
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # type: ignore
except ImportError:
    print("ERROR: SoapySDR python bindings not found. Try: pip3 install SoapySDR (or apt packages).")
    sys.exit(1)

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:
    SMBus = None


STOP = False


def _sigint_handler(signum, frame):
    global STOP
    STOP = True


def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="SoapySDR I/Q logger (CF32 + JSON) + compass heading CSV.")
    p.add_argument("--device", default="driver=bladerf", help="SoapySDR device args string (default: driver=bladerf)")
    p.add_argument("--channel", type=int, default=0, help="RX channel index (default: 0)")
    p.add_argument("--freq", type=float, required=True, help="Center frequency in Hz (e.g. 915e6)")
    p.add_argument("--rate", type=float, required=True, help="Sample rate in S/s (e.g. 5e6)")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB (if supported). If omitted, leaves default.")
    p.add_argument("--bandwidth", type=float, default=None, help="Optional bandwidth in Hz (if supported).")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", default=12.0, type=float, help="Capture duration in seconds (e.g. 3,default 12)")
    group.add_argument("--samples", type=int, help="Capture exactly N complex samples")

    p.add_argument("--out", default="./captures", help="Output directory (default: ./captures)")
    p.add_argument("--name", default=None, help="Base name (default: capture_<UTCtimestamp>)")

    p.add_argument("--block", type=int, default=262144,
                   help="Read block size in complex samples (default: 262144). Larger = fewer syscalls.")
    p.add_argument("--timeout-us", type=int, default=250000,
                   help="Stream read timeout in microseconds (default: 250000)")
    p.add_argument("--settle-ms", type=int, default=150,
                   help="Time to let tuner settle before streaming (default: 150ms)")

    p.add_argument("--tag", default=None, help="Optional tag for metadata (e.g. 'scan1_north_lot')")

    # Compass options (defaults match your script)
    p.add_argument("--compass", action="store_true", help="Enable compass logging (requires smbus2 + sensor present).")
    p.add_argument("--compass-bus", type=int, default=7, help="I2C bus number (default: 7)")
    p.add_argument("--compass-addr", type=lambda x: int(x, 0), default=0x2C, help="I2C address (default: 0x2C)")
    p.add_argument("--compass-hz", type=float, default=30, help="Compass sample rate in Hz (default: 30)")
    p.add_argument("--compass-alpha", type=float, default=0.7, help="EMA smoothing alpha (default: 0.7)")

    # Your calibration defaults
    p.add_argument("--x-offset", type=float, default=149.000)
    p.add_argument("--y-offset", type=float, default=-1451.500)
    p.add_argument("--x-scale", type=float, default=1.026869)
    p.add_argument("--y-scale", type=float, default=0.974501)

    return p.parse_args()


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


TRIG_REGS = [0x09, 0x0A, 0x0B]
TRIG_VAL = 0x01


def trig(bus, addr):
    for r in TRIG_REGS:
        bus.write_byte_data(addr, r, TRIG_VAL)


def read_xyz(bus, addr):
    trig(bus, addr)
    time.sleep(0.001)
    b = bus.read_i2c_block_data(addr, 0x00, 7)
    x = s16(b[1], b[2])
    y = s16(b[3], b[4])
    z = s16(b[5], b[6])
    return b[0], x, y, z


@dataclass
class HeadingState:
    last_heading_deg: float | None = None
    last_status: str = "DISABLED"
    last_sample_monotonic: float | None = None


class CompassThread(threading.Thread):
    def __init__(self, *, bus_num: int, addr: int, hz: float,
                 x_offset: float, y_offset: float, x_scale: float, y_scale: float,
                 alpha: float, csv_path: str, capture_t0_monotonic: float,
                 state: HeadingState, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.bus_num = bus_num
        self.addr = addr
        self.hz = hz
        self.period = 1.0 / hz if hz > 0 else 0.2
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.alpha = alpha
        self.csv_path = csv_path
        self.t0 = capture_t0_monotonic
        self.state = state
        self.stop_event = stop_event

        self._hx_f = None
        self._hy_f = None

    def run(self):
        if SMBus is None:
            self.state.last_status = "NO_SMBUS2"
            return

        try:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            with SMBus(self.bus_num) as bus, open(self.csv_path, "w", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow([
                    "t_rel_s", "status", "heading_deg",
                    "x_raw", "y_raw", "z_raw",
                    "x_cal", "y_cal"
                ])

                self.state.last_status = "OK"

                while not self.stop_event.is_set():
                    t_m = time.monotonic()
                    t_rel = t_m - self.t0

                    try:
                        st, x_raw, y_raw, z_raw = read_xyz(bus, self.addr)

                        # calibrate (your exact calibration)
                        x = (x_raw - self.x_offset) * self.x_scale
                        y = (y_raw - self.y_offset) * self.y_scale

                        # fix rotation direction (your finding)
                        y = -y

                        # unit heading vector
                        theta = math.atan2(y, x)
                        hx = math.cos(theta)
                        hy = math.sin(theta)

                        # EMA on unit vector (circular smoothing)
                        if self._hx_f is None:
                            self._hx_f, self._hy_f = hx, hy
                        else:
                            self._hx_f = self.alpha * self._hx_f + (1 - self.alpha) * hx
                            self._hy_f = self.alpha * self._hy_f + (1 - self.alpha) * hy
                            r = math.hypot(self._hx_f, self._hy_f)
                            if r > 1e-9:
                                self._hx_f /= r
                                self._hy_f /= r

                        heading = math.degrees(math.atan2(self._hy_f, self._hx_f))
                        if heading < 0:
                            heading += 360.0

                        # update shared state
                        self.state.last_heading_deg = float(heading)
                        self.state.last_sample_monotonic = t_m
                        self.state.last_status = f"st=0x{st:02x}"

                        # write CSV row
                        w.writerow([f"{t_rel:.6f}", self.state.last_status, f"{heading:.3f}",
                                    x_raw, y_raw, z_raw, f"{x:.3f}", f"{y:.3f}"])
                        fcsv.flush()

                    except Exception as e:
                        self.state.last_status = f"ERR:{e}"
                        # still log the time, but with blank heading
                        w.writerow([f"{t_rel:.6f}", self.state.last_status, "",
                                    "", "", "", "", ""])
                        fcsv.flush()

                    time.sleep(self.period)

        except Exception as e:
            self.state.last_status = f"THREAD_ERR:{e}"


def main():
    global STOP
    signal.signal(signal.SIGINT, _sigint_handler)

    args = parse_args()
    safe_mkdir(args.out)

    base = args.name or f"capture_{utc_ts_compact()}"
    cf32_path = os.path.join(args.out, base + ".cf32")
    json_path = os.path.join(args.out, base + ".json")
    heading_csv = os.path.join(args.out, base + "_heading.csv")

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
            sdr.setGain(SOAPY_SDR_RX, ch, args.gain)
        except Exception as e:
            print(f"[!] Gain set failed (ignored): {e}")

    # Read back actuals
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
    t_start_wall = time.time()
    t_start_mon = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()

    total_written = 0
    timeouts = 0
    overruns = 0
    read_errors = 0

    # Start compass thread (optional)
    stop_event = threading.Event()
    heading_state = HeadingState()

    compass_thread = None
    if args.compass:
        print(f"[+] Compass logging enabled -> {heading_csv} (bus={args.compass_bus} addr=0x{args.compass_addr:02x})")
        compass_thread = CompassThread(
            bus_num=args.compass_bus,
            addr=args.compass_addr,
            hz=args.compass_hz,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            alpha=args.compass_alpha,
            csv_path=heading_csv,
            capture_t0_monotonic=t_start_mon,
            state=heading_state,
            stop_event=stop_event,
        )
        compass_thread.start()
    else:
        heading_state.last_status = "DISABLED"

    print(f"[+] Writing to: {cf32_path}")
    print(f"[+] Target: {target_samps} complex samples ({target_mode})")
    print("[+] Capturing... Ctrl+C to stop early.")

    with open(cf32_path, "wb", buffering=0) as f:
        while not STOP and total_written < target_samps:
            need = min(block, target_samps - total_written)
            sr = sdr.readStream(rx_stream, [buf], need, timeoutUs=args.timeout_us)

            if sr.ret > 0:
                f.write(buf[:sr.ret].tobytes())
                total_written += sr.ret
                continue

            if sr.ret == 0:
                timeouts += 1
                continue

            read_errors += 1
            try:
                if sr.flags is not None and (sr.flags & SoapySDR.SOAPY_SDR_END_ABRUPT) != 0:
                    overruns += 1
            except Exception:
                pass

            time.sleep(0.001)

    # Stop everything
    stop_event.set()
    if compass_thread is not None:
        compass_thread.join(timeout=1.5)

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    t_end_wall = time.time()
    end_utc = datetime.now(timezone.utc).isoformat()

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
            "elapsed_s": t_end_wall - t_start_wall,
            # This lets you align compass t_rel_s (monotonic-based) with capture start time:
            "t0_monotonic": t_start_mon,
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
            "compass": {
                "enabled": bool(args.compass),
                "bus": args.compass_bus,
                "addr": f"0x{args.compass_addr:02x}",
                "hz": args.compass_hz,
                "alpha": args.compass_alpha,
                "calibration": {
                    "x_offset": args.x_offset,
                    "y_offset": args.y_offset,
                    "x_scale": args.x_scale,
                    "y_scale": args.y_scale,
                },
                "heading_csv": os.path.basename(heading_csv) if args.compass else None,
                "last_heading_deg": heading_state.last_heading_deg,
                "last_status": heading_state.last_status,
            },
            # Reserved for later integration
            "gps": {"lat": None, "lon": None, "alt_m": None, "fix": None},
        },
        "notes": [
            "Raw IQ is complex64 interleaved (I+jQ) at the given sample rate.",
            "Compass CSV timestamps are relative to capture start monotonic time (t_rel_s).",
        ],
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    print("\n[+] Done.")
    print(f"    Samples written: {total_written}")
    print(f"    Elapsed: {t_end_wall - t_start_wall:.3f} s")
    print(f"    Metadata: {json_path}")
    if args.compass:
        print(f"    Heading CSV: {heading_csv}")
    if STOP:
        print("    Stopped early by user (Ctrl+C).")


if __name__ == "__main__":
    main()
