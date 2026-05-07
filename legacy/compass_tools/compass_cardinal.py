#!/usr/bin/env python3
import json
import math
import time
import os
import sys
from smbus2 import SMBus

# Your trigger-based "HMC5883L" clone behavior at 0x2C
TRIG_REGS = [0x09, 0x0A, 0x0B]
TRIG_VAL  = 0x01

DATA_START = 0x00  # status + xyz
N_BYTES    = 7     # [st][xL][xH][yL][yH][zL][zH]

# --------- helpers ---------

def s16(lo: int, hi: int) -> int:
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def wrap360(deg: float) -> float:
    deg %= 360.0
    if deg < 0:
        deg += 360.0
    return deg

def deg_to_cardinal_16(deg: float) -> str:
    # 16-wind compass rose
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    i = int((deg + 11.25) // 22.5) % 16
    return dirs[i]

def load_cal(path: str) -> dict:
    with open(path, "r") as f:
        cal = json.load(f)

    # defaults if missing
    cal.setdefault("bus", 7)
    cal.setdefault("addr", "0x2c")
    cal.setdefault("x_offset", 0.0)
    cal.setdefault("y_offset", 0.0)
    cal.setdefault("x_scale", 1.0)
    cal.setdefault("y_scale", 1.0)
    cal.setdefault("heading_offset_deg", 0.0)
    cal.setdefault("flip_y", False)
    cal.setdefault("smoothing_alpha", 0.85)  # 0=no smoothing, 0.9 heavy

    # addr may be string "0x2c"
    if isinstance(cal["addr"], str):
        cal["addr_int"] = int(cal["addr"], 16)
    else:
        cal["addr_int"] = int(cal["addr"])

    return cal

def trig(bus: SMBus, addr: int) -> None:
    for r in TRIG_REGS:
        bus.write_byte_data(addr, r, TRIG_VAL)

def read_raw_xyz(bus: SMBus, addr: int):
    trig(bus, addr)
    time.sleep(0.03)
    b = bus.read_i2c_block_data(addr, DATA_START, N_BYTES)
    st = b[0]
    x = s16(b[1], b[2])
    y = s16(b[3], b[4])
    z = s16(b[5], b[6])
    return st, x, y, z

# --------- main ---------

def main():
    cal_path = sys.argv[1] if len(sys.argv) > 1 else "compass_cal.json"
    if not os.path.exists(cal_path):
        print(f"Calibration file not found: {cal_path}")
        print("Create compass_cal.json (see instructions).")
        sys.exit(1)

    cal = load_cal(cal_path)

    bus_num = int(cal["bus"])
    addr    = int(cal["addr_int"])
    x_off   = float(cal["x_offset"])
    y_off   = float(cal["y_offset"])
    x_sc    = float(cal["x_scale"])
    y_sc    = float(cal["y_scale"])
    hdg_off = float(cal["heading_offset_deg"])
    flip_y  = bool(cal["flip_y"])
    alpha   = float(cal["smoothing_alpha"])

    # circular smoothing via unit vector EMA
    hx_f = None
    hy_f = None

    print(f"Compass live: bus={bus_num} addr=0x{addr:02x} cal={cal_path}")
    print("Twist the compass. Ctrl+C to exit.\n")

    with SMBus(bus_num) as bus:
        while True:
            st, x_raw, y_raw, z_raw = read_raw_xyz(bus, addr)

            # calibrate
            x = (x_raw - x_off) * x_sc
            y = (y_raw - y_off) * y_sc

            # orientation fix if needed
            if flip_y:
                y = -y

            # compute heading (magnetic)
            theta = math.atan2(y, x)  # radians
            hx = math.cos(theta)
            hy = math.sin(theta)

            # EMA on unit vector
            if hx_f is None:
                hx_f, hy_f = hx, hy
            else:
                hx_f = alpha * hx_f + (1.0 - alpha) * hx
                hy_f = alpha * hy_f + (1.0 - alpha) * hy
                r = math.hypot(hx_f, hy_f)
                if r > 1e-9:
                    hx_f /= r
                    hy_f /= r

            heading = math.degrees(math.atan2(hy_f, hx_f))
            heading = wrap360(heading + hdg_off)

            card = deg_to_cardinal_16(heading)

            # nice one-line output
            sys.stdout.write(
                f"\r{card:>3}  {heading:7.2f}°   "
                f"x={x:8.1f} y={y:8.1f} z={z_raw:6d} st=0x{st:02x}   "
            )
            sys.stdout.flush()

            time.sleep(0.10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
