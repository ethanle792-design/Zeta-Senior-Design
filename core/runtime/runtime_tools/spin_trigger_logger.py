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
# =========================
SPIN_TARGET_DEG = 360        # 1 full rotation
TIME_WINDOW = 30.0           # seconds
TRIGGER_DELAY = 3.0          # seconds before logging
IQ_DURATION = 36             # seconds

# =========================
# COMPASS CONFIG
# =========================
BUS = 7
ADDR = 0x2C

X_OFFSET = 55.5
Y_OFFSET = -1403.5
X_SCALE  = 1.004128
Y_SCALE  = 0.995906

ALPHA = 0.7

TRIG_REGS = [0x09, 0x0A, 0x0B]
TRIG_VAL  = 0x01
# =========================


def build_iq_logger_cmd(capture_num, trigger_mono_ns):
    return [
        "python3",
        "iq_logger_final.py",
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


def trig(bus):
    for r in TRIG_REGS:
        bus.write_byte_data(ADDR, r, TRIG_VAL)


def read_xyz(bus):
    trig(bus)
    time.sleep(0.03)
    b = bus.read_i2c_block_data(ADDR, 0x00, 7)
    x = s16(b[1], b[2])
    y = s16(b[3], b[4])
    z = s16(b[5], b[6])
    return x, y, z


def get_heading(bus, state):
    x_raw, y_raw, z_raw = read_xyz(bus)

    x = (x_raw - X_OFFSET) * X_SCALE
    y = (y_raw - Y_OFFSET) * Y_SCALE
    y = -y

    theta = math.atan2(y, x)
    hx = math.cos(theta)
    hy = math.sin(theta)

    if state["hx"] is None:
        state["hx"], state["hy"] = hx, hy
    else:
        a = ALPHA
        state["hx"] = a * state["hx"] + (1 - a) * hx
        state["hy"] = a * state["hy"] + (1 - a) * hy

        r = math.hypot(state["hx"], state["hy"])
        if r > 1e-9:
            state["hx"] /= r
            state["hy"] /= r

    heading = math.degrees(math.atan2(state["hy"], state["hx"])) - 90
    if heading < 0:
        heading += 360

    return heading


def angle_diff(a, b):
    d = b - a
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def main():
    print(f"[+] Waiting for {SPIN_TARGET_DEG}° spin trigger...")

    history = deque()
    state = {"hx": None, "hy": None}
    capture_num = 0

    with SMBus(BUS) as bus:
        prev_heading = get_heading(bus, state)
        cumulative = 0

        while True:
            now = time.time()
            heading = get_heading(bus, state)

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
                    prev_heading = get_heading(bus, state)
                    continue

            prev_heading = heading
            time.sleep(0.05)


if __name__ == "__main__":
    main()