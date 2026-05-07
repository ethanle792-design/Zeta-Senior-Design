import math, time
from smbus2 import SMBus

BUS = 7
ADDR = 0x2C
TRIG_REGS = [0x09, 0x0A, 0x0B]
TRIG_VAL  = 0x01

X_OFFSET = 149.000
Y_OFFSET = -1451.500
X_SCALE  = 1.026869
Y_SCALE  = 0.974501

# smoothing: 0.0 = no smoothing, 0.9 = heavy smoothing
ALPHA = 0.7

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
    return b[0], x, y, z

def main():
    hx_f = None
    hy_f = None

    with SMBus(BUS) as bus:
        while True:
            st, x_raw, y_raw, z_raw = read_xyz(bus)

            # calibrate
            x = (x_raw - X_OFFSET) * X_SCALE
            y = (y_raw - Y_OFFSET) * Y_SCALE

            # fix rotation direction (your finding)
            y = -y

            # optional: reject extreme magnitude outliers (tune later)
            m = math.sqrt(x*x + y*y + z_raw*z_raw)
            # if not (100 < m < 2000):
            #     continue

            # convert to unit heading vector
            theta = math.atan2(y, x)
            hx = math.cos(theta)
            hy = math.sin(theta)

            # EMA on the unit vector (circular smoothing)
            if hx_f is None:
                hx_f, hy_f = hx, hy
            else:
                hx_f = ALPHA * hx_f + (1-ALPHA) * hx
                hy_f = ALPHA * hy_f + (1-ALPHA) * hy
                # renormalize
                r = math.hypot(hx_f, hy_f)
                if r > 1e-9:
                    hx_f /= r
                    hy_f /= r

            heading = math.degrees(math.atan2(hy_f, hx_f))
            if heading < 0:
                heading += 360.0

            print(f"st=0x{st:02x} x_raw={x_raw:6d} y_raw={y_raw:6d} "
                  f"x={x:7.1f} y={y:7.1f} z={z_raw:6d} hdg={heading:7.2f}°")

            time.sleep(0.2)

if __name__ == "__main__":
    main()
