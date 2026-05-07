#!/usr/bin/env python3
import argparse
import math
import signal
import sys
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from smbus2 import SMBus
    HAS_I2C = True
except ImportError:
    HAS_I2C = False


# ============================================================
# Compass logic adapted from compass_cardinal_actual.py
# with offsets hard-coded from latest compass_cal.json
# ============================================================

BUS = 7
ADDR = 0x2C

TRIG_REGS = [0x09, 0x0A, 0x0B]
TRIG_VAL = 0x01

DATA_START = 0x00
N_BYTES = 7  # [st][xL][xH][yL][yH][zL][zH]

X_OFFSET = 55.5
Y_OFFSET = -1403.5
X_SCALE = 1.004128
Y_SCALE = 0.995906
HEADING_OFFSET_DEG = 0.0
SMOOTHING_ALPHA = 0.7
FLIP_Y = True


def s16(lo: int, hi: int) -> int:
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def wrap360(deg: float) -> float:
    deg %= 360.0
    if deg < 0:
        deg += 360.0
    return deg


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


class CompassReader:
    def __init__(self, bus_num=BUS, addr=ADDR):
        if not HAS_I2C:
            raise RuntimeError("smbus2 is not installed")

        self.bus = SMBus(bus_num)
        self.addr = addr
        self.hx_f = None
        self.hy_f = None

    def read_heading(self):
        st, x_raw, y_raw, z_raw = read_raw_xyz(self.bus, self.addr)

        x = (x_raw - X_OFFSET) * X_SCALE
        y = (y_raw - Y_OFFSET) * Y_SCALE

        if FLIP_Y:
            y = -y

        theta = math.atan2(y, x)
        hx = math.cos(theta)
        hy = math.sin(theta)

        if self.hx_f is None:
            self.hx_f, self.hy_f = hx, hy
        else:
            a = SMOOTHING_ALPHA
            self.hx_f = a * self.hx_f + (1.0 - a) * hx
            self.hy_f = a * self.hy_f + (1.0 - a) * hy
            r = math.hypot(self.hx_f, self.hy_f)
            if r > 1e-9:
                self.hx_f /= r
                self.hy_f /= r

        # matches compass_cardinal_actual.py
        heading = math.degrees(math.atan2(self.hy_f, self.hx_f)) - 90
        heading = wrap360(heading + HEADING_OFFSET_DEG)

        return {
            "heading": heading,
            "status": st,
            "x": x,
            "y": y,
            "z": z_raw,
            "x_raw": x_raw,
            "y_raw": y_raw,
        }

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


# ============================================================
# SDR helpers
# ============================================================

def kw_to_dict(k):
    try:
        return dict(k)
    except Exception:
        try:
            return {key: k[key] for key in k.keys()}
        except Exception:
            return {}


def pick_device(args):
    devs = SoapySDR.Device.enumerate()
    print(f"Devices found: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"  [{i}] {d}")

    if not devs:
        raise RuntimeError("No SoapySDR devices found.")

    if args.index is not None:
        if args.index < 0 or args.index >= len(devs):
            raise RuntimeError(f"--index {args.index} out of range")
        return devs[args.index]

    want_driver = (args.driver or "").lower()
    want_serial = (args.serial or "").strip()

    for d in devs:
        dd = kw_to_dict(d)
        driver = str(dd.get("driver", "")).lower()
        serial = str(dd.get("serial", "")).strip()

        if want_driver and driver != want_driver:
            continue
        if want_serial and serial != want_serial:
            continue
        return d

    if want_serial or want_driver:
        raise RuntimeError(
            f"Requested device not found (driver={args.driver!r}, serial={args.serial!r})"
        )

    for d in devs:
        dd = kw_to_dict(d)
        if str(dd.get("driver", "")).lower() == "bladerf":
            return d

    return devs[0]


def set_rx_params(sdr, args):
    sr = float(args.rate)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sr)

    cf = float(args.freq)
    sdr.setFrequency(SOAPY_SDR_RX, 0, cf)

    if args.bw is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, 0, float(args.bw))
        except Exception:
            print("[warn] setBandwidth failed; continuing.")

    if args.gain is not None:
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, float(args.gain))
        except Exception:
            print("[warn] setGain failed; continuing.")

    try:
        actual_cf = sdr.getFrequency(SOAPY_SDR_RX, 0)
    except Exception:
        actual_cf = cf

    try:
        actual_sr = sdr.getSampleRate(SOAPY_SDR_RX, 0)
    except Exception:
        actual_sr = sr

    print("\n[rx] Config:")
    print(f"  Center freq: {actual_cf/1e6:.6f} MHz")
    print(f"  Sample rate: {actual_sr/1e6:.6f} Msps")
    if args.gain is not None:
        print(f"  Gain:        {args.gain:.1f} dB")
    print("")

    return actual_cf, actual_sr


# ============================================================
# Main live display
# ============================================================

def run_stream_test(sdr, args, cf, sr):
    print("[stream] Setting up RX stream...")
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
    sdr.activateStream(rx_stream)

    mtu = 0
    try:
        mtu = int(sdr.getStreamMTU(rx_stream))
    except Exception:
        mtu = 8192

    buf_len = int(args.buf) if args.buf else max(4096, min(mtu, 16384))
    buff = np.empty(buf_len, np.complex64)

    print(f"[stream] MTU={mtu} using buf_len={buf_len}")
    print(f"[stream] Running for {args.seconds:.1f}s (Ctrl+C to stop)...\n")

    start = time.time()
    last_report = start
    samples_total = 0

    overruns = 0
    timeouts = 0
    other_errs = 0

    do_plot = bool(args.plot)
    do_compass = bool(args.compass)

    waterfall = None
    fig = None
    ax_fft = None
    ax_wf = None
    line = None
    im = None

    compass_reader = None
    compass_fig = None
    compass_ax = None
    compass_arrow = None
    compass_text = None

    if do_compass:
        compass_reader = CompassReader()

    if do_plot:
        if plt is None:
            raise RuntimeError("matplotlib is not installed")

        plt.ion()

        N = int(args.fft)
        freqs = cf + np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / sr))
        freqs_mhz = freqs / 1e6

        fig, (ax_fft, ax_wf) = plt.subplots(2, 1, figsize=(12, 8))
        line, = ax_fft.plot(freqs_mhz, np.zeros(N, dtype=float))
        ax_fft.set_title("Live Spectrum")
        ax_fft.set_xlabel("Frequency (MHz)")
        ax_fft.set_ylabel("Magnitude (dB)")
        ax_fft.grid(True)

        waterfall = np.full((args.waterfall_rows, N), -120.0, dtype=np.float32)
        im = ax_wf.imshow(
            waterfall,
            aspect="auto",
            origin="lower",
            extent=[freqs_mhz[0], freqs_mhz[-1], 0, args.waterfall_rows]
        )
        ax_wf.set_title("Waterfall")
        ax_wf.set_xlabel("Frequency (MHz)")
        ax_wf.set_ylabel("Time")

        fig.canvas.draw()
        fig.canvas.flush_events()

        if do_compass and args.separate_compass_window:
            compass_fig, compass_ax = plt.subplots(figsize=(5, 5))
            compass_ax.set_title("Compass Heading")
            compass_ax.set_xlim(-1.2, 1.2)
            compass_ax.set_ylim(-1.2, 1.2)
            compass_ax.set_aspect("equal")
            compass_ax.grid(True)

            circle = plt.Circle((0, 0), 1.0, fill=False)
            compass_ax.add_patch(circle)

            compass_ax.text(0, 1.08, "N", ha="center", va="center", fontsize=12)
            compass_ax.text(1.08, 0, "E", ha="center", va="center", fontsize=12)
            compass_ax.text(0, -1.08, "S", ha="center", va="center", fontsize=12)
            compass_ax.text(-1.08, 0, "W", ha="center", va="center", fontsize=12)

            compass_arrow, = compass_ax.plot([0, 0], [0, 1], linewidth=3)
            compass_text = compass_ax.text(0, -1.25, "Heading: ---.-°", ha="center", fontsize=14)

    stop_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            now = time.time()
            if stop_flag["stop"]:
                break
            if args.seconds > 0 and (now - start) >= args.seconds:
                break

            sr_ret = sdr.readStream(rx_stream, [buff], len(buff), timeoutUs=int(args.timeout_us))

            heading_info = None
            if compass_reader is not None:
                try:
                    heading_info = compass_reader.read_heading()
                except Exception as e:
                    heading_info = {"error": str(e)}

            if sr_ret.ret > 0:
                n = sr_ret.ret
                iq = buff[:n]
                samples_total += n

                if do_plot:
                    N = int(args.fft)
                    if n >= N:
                        x = iq[:N]
                    else:
                        x = np.zeros(N, np.complex64)
                        x[:n] = iq

                    spec = np.fft.fftshift(np.fft.fft(x, n=N))
                    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)

                    line.set_ydata(mag_db)
                    ax_fft.set_xlim(freqs_mhz[0], freqs_mhz[-1])

                    ymin = float(np.min(mag_db))
                    ymax = float(np.max(mag_db))
                    if ymax - ymin < 10:
                        ymax = ymin + 10
                    ax_fft.set_ylim(ymin - 5, ymax + 5)

                    waterfall = np.roll(waterfall, -1, axis=0)
                    waterfall[-1, :] = mag_db
                    im.set_data(waterfall)
                    im.set_clim(np.percentile(waterfall, 5), np.percentile(waterfall, 95))

                    if heading_info and "heading" in heading_info and not args.separate_compass_window:
                        ax_fft.set_title(f"Live Spectrum | Heading: {heading_info['heading']:.1f}°")
                    else:
                        ax_fft.set_title("Live Spectrum")

                    if heading_info and "heading" in heading_info and args.separate_compass_window and compass_arrow is not None:
                        heading = heading_info["heading"]
                        rad = math.radians(heading)
                        px = math.sin(rad)
                        py = math.cos(rad)
                        compass_arrow.set_data([0, px], [0, py])

                        if compass_text is not None:
                            compass_text.set_text(
                                f"Heading: {heading:.1f}°   x={heading_info['x']:.1f} y={heading_info['y']:.1f} z={heading_info['z']}"
                            )

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    if compass_fig is not None:
                        compass_fig.canvas.draw()
                        compass_fig.canvas.flush_events()

            else:
                if sr_ret.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                    timeouts += 1
                elif sr_ret.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                    overruns += 1
                else:
                    other_errs += 1

            if heading_info and "heading" in heading_info:
                sys.stdout.write(
                    f"\rHeading: {heading_info['heading']:7.2f}°   "
                    f"x={heading_info['x']:8.1f} y={heading_info['y']:8.1f} "
                    f"z={heading_info['z']:6d} st=0x{heading_info['status']:02x}   "
                )
                sys.stdout.flush()
            elif heading_info and "error" in heading_info:
                sys.stdout.write(f"\rCompass error: {heading_info['error']}   ")
                sys.stdout.flush()

            if (now - last_report) >= args.report_s:
                elapsed = now - start
                rate_eff = (samples_total / elapsed) if elapsed > 0 else 0.0

                print(
                    f"\nt={elapsed:6.1f}s  eff_rate={rate_eff/1e6:6.3f} Msps  "
                    f"overruns={overruns}  timeouts={timeouts}  other={other_errs}"
                )
                last_report = now

            if do_plot:
                plt.pause(0.001)

    finally:
        print("\n[stream] Stopping...")
        try:
            sdr.deactivateStream(rx_stream)
        except Exception:
            pass
        try:
            sdr.closeStream(rx_stream)
        except Exception:
            pass
        if compass_reader is not None:
            compass_reader.close()

    elapsed = max(1e-9, time.time() - start)
    print("\n[summary]")
    print(f"  Duration:    {elapsed:.2f} s")
    print(f"  Samples:     {samples_total}")
    print(f"  Eff rate:    {samples_total/elapsed/1e6:.3f} Msps")
    print(f"  Overruns:    {overruns}")
    print(f"  Timeouts:    {timeouts}")
    print(f"  Other errs:  {other_errs}")


def parse_args():
    p = argparse.ArgumentParser(description="SoapySDR bladeRF RX test with optional waterfall + compass.")
    p.add_argument("--driver", default="bladerf", help="Soapy driver to select")
    p.add_argument("--serial", default="", help="Optional SDR serial to select")
    p.add_argument("--index", type=int, default=None, help="Device index from enumerate() to open")

    p.add_argument("--freq", type=float, default=100e6, help="Center frequency (Hz)")
    p.add_argument("--rate", type=float, default=2e6, help="Sample rate (S/s)")
    p.add_argument("--bw", type=float, default=None, help="Analog bandwidth (Hz)")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB")
    p.add_argument("--seconds", type=float, default=10.0, help="Run time in seconds, 0 = forever")
    p.add_argument("--timeout-us", type=int, default=200000, help="readStream timeout (microseconds)")
    p.add_argument("--report-s", type=float, default=1.0, help="Status print interval (s)")

    p.add_argument("--buf", type=int, default=0, help="RX buffer length in samples (0=auto)")
    p.add_argument("--plot", action="store_true", help="Show live spectrum + waterfall")
    p.add_argument("--fft", type=int, default=4096, help="FFT size for plotting")
    p.add_argument("--waterfall-rows", type=int, default=200, help="Number of waterfall rows")

    p.add_argument("--compass", action="store_true", help="Enable compass heading display")
    p.add_argument("--separate-compass-window", action="store_true", help="Show compass in a separate window")

    return p.parse_args()


def main():
    args = parse_args()

    print("=== SDR Live Test (SoapySDR + Compass) ===")
    print(f"Driver:      {args.driver}")
    if args.serial:
        print(f"Serial:      {args.serial}")
    if args.index is not None:
        print(f"Index:       {args.index}")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    if args.bw is not None:
        print(f"Bandwidth:   {args.bw/1e6:.3f} MHz")
    if args.gain is not None:
        print(f"Gain:        {args.gain:.1f} dB")
    print(f"Seconds:     {args.seconds:.1f}")
    print(f"Plot:        {args.plot}")
    print(f"Compass:     {args.compass}")
    print("")

    picked = pick_device(args)
    print(f"[open] {picked}")
    sdr = SoapySDR.Device(picked)

    cf, sr = set_rx_params(sdr, args)
    run_stream_test(sdr, args, cf, sr)


if __name__ == "__main__":
    main()