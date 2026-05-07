#!/usr/bin/env python3
import time
import math
import argparse
from smbus2 import SMBus

# --- BNO055 registers ---
BNO055_CHIP_ID_ADDR       = 0x00
BNO055_PAGE_ID_ADDR       = 0x07
BNO055_EUL_HEADING_LSB    = 0x1A
BNO055_CALIB_STAT_ADDR    = 0x35
BNO055_OPR_MODE_ADDR      = 0x3D
BNO055_PWR_MODE_ADDR      = 0x3E
BNO055_SYS_TRIGGER_ADDR   = 0x3F

# --- Expected values ---
BNO055_ID                 = 0xA0

# --- Power modes ---
POWER_MODE_NORMAL         = 0x00

# --- Operation modes ---
OPERATION_MODE_CONFIG     = 0x00
OPERATION_MODE_COMPASS    = 0x09
OPERATION_MODE_NDOF       = 0x0C


def wrap360(deg: float) -> float:
    deg %= 360.0
    if deg < 0:
        deg += 360.0
    return deg


def deg_to_cardinal_16(deg: float) -> str:
    dirs = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    idx = int((deg + 11.25) // 22.5) % 16
    return dirs[idx]


def s16(lsb: int, msb: int) -> int:
    value = (msb << 8) | lsb
    if value & 0x8000:
        value -= 65536
    return value


class BNO055:
    def __init__(self, bus_num: int = 7, addr: int = 0x28):
        self.bus_num = bus_num
        self.addr = addr
        self.bus = SMBus(bus_num)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def read_u8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.addr, reg)

    def write_u8(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.addr, reg, value & 0xFF)

    def read_len(self, reg: int, n: int):
        return self.bus.read_i2c_block_data(self.addr, reg, n)

    def check_chip_id(self) -> bool:
        chip_id = self.read_u8(BNO055_CHIP_ID_ADDR)
        return chip_id == BNO055_ID

    def set_mode(self, mode: int) -> None:
        self.write_u8(BNO055_OPR_MODE_ADDR, mode)
        time.sleep(0.03)

    def initialize(self, use_ndof: bool = True) -> None:
        chip_id = self.read_u8(BNO055_CHIP_ID_ADDR)
        if chip_id != BNO055_ID:
            raise RuntimeError(
                f"BNO055 not found at 0x{self.addr:02X} on bus {self.bus_num}. "
                f"Read chip ID 0x{chip_id:02X}, expected 0x{BNO055_ID:02X}."
            )

        self.set_mode(OPERATION_MODE_CONFIG)
        self.write_u8(BNO055_PAGE_ID_ADDR, 0x00)
        self.write_u8(BNO055_PWR_MODE_ADDR, POWER_MODE_NORMAL)
        time.sleep(0.01)

        # Optional external crystal enable.
        # Usually safe on Adafruit boards. If it causes issues, comment it out.
        self.write_u8(BNO055_SYS_TRIGGER_ADDR, 0x80)
        time.sleep(0.7)

        self.set_mode(OPERATION_MODE_NDOF if use_ndof else OPERATION_MODE_COMPASS)
        time.sleep(0.05)

    def read_heading_deg(self) -> float:
        data = self.read_len(BNO055_EUL_HEADING_LSB, 2)
        raw = s16(data[0], data[1])
        heading = raw / 16.0
        return wrap360(heading)

    def read_calib_status(self):
        calib = self.read_u8(BNO055_CALIB_STAT_ADDR)
        sys_cal = (calib >> 6) & 0x03
        gyro_cal = (calib >> 4) & 0x03
        accel_cal = (calib >> 2) & 0x03
        mag_cal = calib & 0x03
        return sys_cal, gyro_cal, accel_cal, mag_cal


def main():
    parser = argparse.ArgumentParser(description="BNO055 CLI heading display")
    parser.add_argument("--bus", type=int, default=7, help="I2C bus number (default: 7)")
    parser.add_argument("--addr", type=lambda x: int(x, 0), default=0x28,
                        help="I2C address (default: 0x28)")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Update rate in Hz (default: 10)")
    parser.add_argument("--compass-mode", action="store_true",
                        help="Use COMPASS mode instead of NDOF")
    parser.add_argument("--heading-offset", type=float, default=180.0,
                        help="Fixed boresight offset in degrees (default: 180.0)")
    args = parser.parse_args()

    period = 1.0 / max(args.hz, 0.1)

    bno = BNO055(bus_num=args.bus, addr=args.addr)
    try:
        bno.initialize(use_ndof=not args.compass_mode)

        mode_name = "COMPASS" if args.compass_mode else "NDOF"
        print(f"BNO055 online: bus={args.bus} addr=0x{args.addr:02X} mode={mode_name}")
        print(f"Boresight offset: {args.heading_offset:.2f}°")
        print("Move the sensor around to improve calibration. Ctrl+C to exit.\n")

        while True:
            raw_heading = bno.read_heading_deg()
            heading = wrap360(raw_heading + args.heading_offset)
            card = deg_to_cardinal_16(heading)
            sys_cal, gyro_cal, accel_cal, mag_cal = bno.read_calib_status()

            line = (
                f"\r{card:>3}  {heading:7.2f}°   "
                f"(raw {raw_heading:7.2f}°)   "
                f"CAL SYS:{sys_cal} G:{gyro_cal} A:{accel_cal} M:{mag_cal}   "
            )
            print(line, end="", flush=True)

            time.sleep(period)

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        bno.close()


if __name__ == "__main__":
    main()