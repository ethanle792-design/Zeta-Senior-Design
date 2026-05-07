#!/usr/bin/env python3

import math
import time
import subprocess
from collections import deque
from smbus2 import SMBus
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./captures")
    return p.parse_args()


args = parse_args()

# =========================
# USER CONFIG
# =========================x``
SPIN_TARGET_DEG = 360        # 1 full rotation
TIME_WINDOW = 30.0           # seconds
TRIGGER_DELAY = 3.0          # seconds before logging
IQ_DURATION = 36             # seconds

# =========================
# COMPASS CONFIG
# =========================
BUS = 7
ADDR = 0x28
# =========================


def build_iq_logger_cmd(capture_num, trigger_mono_ns):
    return [
        "python3",
        "iq_logger_newCompass.py",
        "--freq", "915.4e6",
        "--rate", "0.6e6",
        "--duration", str(IQ_DURATION),
        "--compass",
        "--out", args.out,
        "--name", f"spin_capture_{capture_num}" ,
        "--trigger-mono-ns", str(trigger_mono_ns)
    ]


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def angle_diff(a, b):
    d = b - a
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d

def wrap360(deg):
    deg %= 360.0
    if deg < 0:
        deg += 360.0
    return deg


def get_heading(bus):
    # force page 0
    bus.write_byte_data(ADDR, 0x07, 0x00)
    time.sleep(0.001)

    data = bus.read_i2c_block_data(ADDR, 0x1A, 2)

    raw = s16(data[0], data[1])
    heading = raw / 16.0

    # Apply boresight offset
    heading = wrap360(heading + 180.0)

    return heading

def main():
    print(f"[+] Waiting for {SPIN_TARGET_DEG}° spin trigger...")

    history = deque()
    capture_num = 0

    with SMBus(BUS) as bus:
        prev_heading = get_heading(bus)
        cumulative = 0

        while True:
            now = time.time()
            heading = get_heading(bus)

            d = angle_diff(prev_heading, heading)
            if abs(d) < 2.0:
                d = 0.0
            
            cumulative += d

            history.append((now, cumulative))

            while history and (now - history[0][0]) > TIME_WINDOW:
                history.popleft()

            if len(history) >= 2:
                rotation_window = cumulative - history[0][1]
                print(f"\rNet Rotation: {rotation_window:.1f}°", end="", flush=True)

                if abs(rotation_window) >= SPIN_TARGET_DEG:
                    print("\n[+] Spin detected! Triggering capture...")

                    trigger_mono_ns = time.monotonic_ns()

                    time.sleep(TRIGGER_DELAY)

                    cmd = build_iq_logger_cmd(capture_num, trigger_mono_ns)
                    print("[+] Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)

                    print("[+] Capture complete. Resetting...")

                    capture_num += 1
                    history.clear()
                    cumulative = 0
                    prev_heading = get_heading(bus)
                    continue

            prev_heading = heading
            time.sleep(0.05)


if __name__ == "__main__":
    main()