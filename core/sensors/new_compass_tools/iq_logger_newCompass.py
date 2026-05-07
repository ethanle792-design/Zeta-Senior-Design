#!/usr/bin/env python3
"""
iq_logger_compass_cs16.py

Same behavior as iq_logger_compass.py but stores IQ as CS16
(interleaved int16 I/Q) instead of CF32.

Output files:
  <base>.cs16
  <base>.json
  <base>_heading.csv
"""

import argparse
import csv
import json
import os
import signal
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
from smbus2 import SMBus


STOP = False


def _sigint_handler(signum, frame):
    global STOP
    STOP = True


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

def mono_ns():
    return time.monotonic_ns()


def mkdir(path):
    os.makedirs(path, exist_ok=True)

def deg_to_cardinal_16(deg: float) -> str:
    # 16-wind compass rose
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    i = int((deg + 11.25) // 22.5) % 16
    return dirs[i]

def wrap360(deg: float) -> float:
    deg %= 360.0
    if deg < 0:
        deg += 360.0
    return deg


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="driver=bladerf")
    p.add_argument("--channel", type=int, default=0)

    p.add_argument("--freq", type=float, default=915.4e6)
    p.add_argument("--rate", type=float, default=0.6e6)
    p.add_argument("--gain", type=float, default=0)
    p.add_argument("--bandwidth", type=float)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", type=float)
    group.add_argument("--samples", type=int)

    p.add_argument("--out", default="./captures")
    p.add_argument("--name")

    p.add_argument("--block", type=int, default=262144)
    p.add_argument("--timeout-us", type=int, default=250000)

    p.add_argument("--compass", action="store_true")
    p.add_argument("--compass-bus", type=int, default=7)
    p.add_argument("--compass-addr", type=lambda x: int(x, 0), default=0x28)
    p.add_argument("--compass-hz", type=float, default=30)

    # Hard-coded defaults from latest compass_cal.json
    p.add_argument("--compass-alpha", type=float, default=0.7)
    p.add_argument("--trigger-mono-ns", type=int, default=None)

    p.add_argument("--heading-offset", type=float, default=180.0)

    return p.parse_args()


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v



@dataclass
class HeadingState:
    heading: float | None = None
    status: str = "DISABLED"

class CompassThread(threading.Thread):

    def __init__(self, args, csv_path, state, start_t, stop_event):
        super().__init__(daemon=True)

        self.args = args
        self.csv_path = csv_path
        self.state = state
        self.t0 = start_t
        self.stop_event = stop_event
        self.last_heading = None

    def run(self):
        with SMBus(self.args.compass_bus) as bus, open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "status", "heading", "cardinal_direction"])

            period = 1 / self.args.compass_hz

            while not self.stop_event.is_set():
                t = time.monotonic() - self.t0

                try:
                    bus.write_byte_data(self.args.compass_addr, 0x07, 0x00)
                    time.sleep(0.001)

                    # --- Read BNO055 heading ---
                    data = bus.read_i2c_block_data(self.args.compass_addr, 0x1A, 2)

                    raw = s16(data[0], data[1])
                    heading = raw / 16.0

                    # --- Apply boresight offset (180° fix) ---
                    heading = wrap360(heading + self.args.heading_offset)

                    self.state.heading = heading
                    self.last_heading = heading
                    self.state.status = "OK"

                    w.writerow([
                        f"{t:.4f}",
                        self.state.status,
                        f"{heading:.2f}",
                        deg_to_cardinal_16(heading)
                    ])

                except Exception as e:
                    self.state.status = str(e)
                    if self.last_heading is not None:
                        self.state.heading = self.last_heading

                    w.writerow([f"{t:.4f}", "ERR", "", ""])


                f.flush()
                time.sleep(period)

def main():
    global STOP
    signal.signal(signal.SIGINT, _sigint_handler)

    args = parse_args()

    mkdir(args.out)

    base = args.name or f"capture_{utc_ts()}"

    iq_path = os.path.join(args.out, base + ".cs16")
    json_path = os.path.join(args.out, base + ".json")
    heading_csv = os.path.join(args.out, base + "_heading.csv")

    print("[+] Opening SDR")

    sdr = SoapySDR.Device(args.device)

    ch = args.channel

    sdr.setSampleRate(SOAPY_SDR_RX, ch, args.rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, args.freq)

    if args.bandwidth is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, ch, args.bandwidth)
        except Exception:
            pass

    if args.gain is not None:
        sdr.setGain(SOAPY_SDR_RX, ch, args.gain)

    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
    sdr.activateStream(rx_stream)

    buf = np.empty(args.block, np.complex64)

    if args.samples:
        target = args.samples
    else:
        target = int(args.duration * args.rate)

    written = 0

    state = HeadingState()
    stop_event = threading.Event()

    ct = None
    compass_start_mono_ns = None
    iq_start_mono_ns = None   

    if args.compass:
        compass_start_mono_ns = mono_ns()
        
        ct = CompassThread(args, heading_csv, state, time.monotonic(), stop_event)
        ct.start()

    with open(iq_path, "wb", buffering=0) as f:
        first_iq_block = True

        while not STOP and written < target:
            need = min(args.block, target - written)

            sr = sdr.readStream(rx_stream, [buf], need, timeoutUs=args.timeout_us)

            if sr.ret > 0:
                if first_iq_block:
                    iq_start_mono_ns = mono_ns()
                    first_iq_block = False

                iq = buf[:sr.ret]

                i = np.clip(iq.real * 32767, -32768, 32767).astype(np.int16)
                q = np.clip(iq.imag * 32767, -32768, 32767).astype(np.int16)

                inter = np.empty(i.size * 2, np.int16)
                inter[0::2] = i
                inter[1::2] = q

                f.write(inter.tobytes())
                written += sr.ret

    stop_event.set()
    if ct is not None:
        ct.join(timeout=1.0)

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    meta = {
        "IQ_file": base + ".cs16",
        "heading_file": base + "_heading.csv" if args.compass else None,
        "duration_s": args.duration,
        "requested_samples": args.samples,
        "gps": None,
        "log_delta_ns": (
                iq_start_mono_ns - compass_start_mono_ns
                if args.compass and iq_start_mono_ns is not None
                else None
            ),
        "trigger_to_iq_start_ns": (
                iq_start_mono_ns - args.trigger_mono_ns
                if args.trigger_mono_ns is not None and iq_start_mono_ns is not None
                else None
            ),
        "samples": written,
        "rate": args.rate,
        "freq": args.freq,
        "heading_final": state.heading,
        "compass_enabled": args.compass,
        "heading_correction_deg": args.heading_offset,
    }

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done")
    print("Samples:", written)
    if args.compass:
        print("Final heading:", state.heading)
        print("Heading CSV:", heading_csv)


if __name__ == "__main__":
    main()