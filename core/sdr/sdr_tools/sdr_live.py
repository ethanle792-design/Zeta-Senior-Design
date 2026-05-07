#!/usr/bin/env python3
"""
sdr_live.py (BLADErf/SoapySDR) — simple, robust SDR-only test (NO GPS/compass)

What it does:
- Enumerates SoapySDR devices
- Selects bladeRF (by --serial, --index, or first driver=bladerf)
- Configures Rx (freq, sample-rate, bandwidth, gain)
- Streams IQ and prints basic power stats (dBFS-like) + overrun/timeout counters
- Optional: live spectrum plot with --plot (matplotlib)

Dependencies:
  sudo apt install -y python3-soapysdr python3-numpy python3-matplotlib soapysdr-tools
"""

import argparse
import math
import sys
import time
import signal

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# Optional GUI import is delayed so headless runs don't crash.


def kw_to_dict(k):
    """Convert SoapySDRKwargs-ish object to a real dict safely."""
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
        raise RuntimeError("No SoapySDR devices found. (SoapySDRUtil --find should show something.)")

    # Explicit index wins
    if args.index is not None:
        if args.index < 0 or args.index >= len(devs):
            raise RuntimeError(f"--index {args.index} out of range (0..{len(devs)-1})")
        print(f"[pick] Using --index {args.index}")
        return devs[args.index]

    # Otherwise pick by driver/serial
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

    # If they asked for a specific serial/driver and we didn't find it
    if want_serial or want_driver:
        raise RuntimeError(f"Requested device not found (driver={args.driver!r}, serial={args.serial!r}).")

    # Default: first bladerf, else dev[0]
    for d in devs:
        dd = kw_to_dict(d)
        if str(dd.get("driver", "")).lower() == "bladerf":
            return d

    print("[pick] No driver=bladerf found; falling back to device[0].")
    return devs[0]


def list_gain_elements(sdr):
    """Print gain elements if the driver supports them."""
    try:
        gains = sdr.listGains(SOAPY_SDR_RX, 0)
        if gains:
            print(f"[gain] Elements: {gains}")
            for g in gains:
                try:
                    r = sdr.getGainRange(SOAPY_SDR_RX, 0, g)
                    print(f"       {g}: range {r.minimum()}..{r.maximum()} step={r.step()}")
                except Exception:
                    pass
        else:
            print("[gain] No gain elements reported (driver may use a single gain).")
    except Exception:
        print("[gain] listGains() not supported by this driver.")


def set_rx_params(sdr, args):
    # Sample rate
    sr = float(args.rate)
    actual_sr = sdr.setSampleRate(SOAPY_SDR_RX, 0, sr)
    # Some drivers return None; query it back
    try:
        actual_sr = sdr.getSampleRate(SOAPY_SDR_RX, 0)
    except Exception:
        actual_sr = sr

    # Center frequency
    cf = float(args.freq)
    try:
        sdr.setFrequency(SOAPY_SDR_RX, 0, cf)
    except Exception as e:
        raise RuntimeError(f"setFrequency failed: {e}")

    try:
        actual_cf = sdr.getFrequency(SOAPY_SDR_RX, 0)
    except Exception:
        actual_cf = cf

    # Bandwidth (optional)
    if args.bw is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, 0, float(args.bw))
        except Exception:
            print("[warn] setBandwidth() not supported or failed; continuing.")
    try:
        actual_bw = sdr.getBandwidth(SOAPY_SDR_RX, 0)
    except Exception:
        actual_bw = None

    # Gain
    if args.gain is not None:
        # Try overall gain first
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, float(args.gain))
        except Exception:
            # If that fails, try a common bladeRF element name sequence
            # (Different Soapy modules expose different names.)
            try:
                for name in ("LNA", "VGA1", "VGA2", "RF", "IF"):
                    try:
                        sdr.setGain(SOAPY_SDR_RX, 0, name, float(args.gain))
                        break
                    except Exception:
                        continue
            except Exception:
                print("[warn] setGain failed; continuing.")

    print("\n[rx] Config:")
    print(f"  Center freq: {actual_cf/1e6:.6f} MHz")
    print(f"  Sample rate: {actual_sr/1e6:.6f} Msps")
    if actual_bw is not None:
        print(f"  Bandwidth:   {actual_bw/1e6:.6f} MHz")
    if args.gain is not None:
        # Query overall gain if possible
        try:
            g = sdr.getGain(SOAPY_SDR_RX, 0)
            print(f"  Gain:        {g:.1f} dB (reported)")
        except Exception:
            print(f"  Gain:        {args.gain:.1f} dB (set)")
    print("")

    return actual_cf, actual_sr


def power_stats(iq: np.ndarray):
    """
    Compute basic power stats.
    Treats full-scale as magnitude 1.0 (CF32 normalized-ish), so results are "dBFS-like".
    """
    p = np.abs(iq) ** 2
    mean_p = float(np.mean(p))
    peak_p = float(np.max(p))
    # Avoid log(0)
    mean_db = 10.0 * math.log10(mean_p + 1e-20)
    peak_db = 10.0 * math.log10(peak_p + 1e-20)
    return mean_db, peak_db


def run_stream_test(sdr, args, cf, sr):
    print("[stream] Setting up RX stream...")
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
    sdr.activateStream(rx_stream)

    mtu = 0
    try:
        mtu = int(sdr.getStreamMTU(rx_stream))
    except Exception:
        mtu = 8192

    # Use a moderate buffer size; too big increases latency, too small increases overhead.
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

    # Simple rolling stats
    mean_db_acc = []
    peak_db_acc = []

    # Optional spectrum plot
    do_plot = bool(args.plot)
    if do_plot:
        import matplotlib
        import matplotlib.pyplot as plt

        plt.ion()
        N = int(args.fft)
        freqs = cf + np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / sr))
        fig, ax = plt.subplots()
        line, = ax.plot(freqs / 1e6, np.zeros_like(freqs, dtype=float))
        ax.set_title("Live Spectrum (SoapySDR bladeRF)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
        ax.set_ylim(-50,100)
        ax.grid(True)
        fig.canvas.draw()
        fig.canvas.flush_events()

    stop_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            now = time.time()
            if stop_flag["stop"]:
                break
            if (now - start) >= args.seconds:
                break

            sr_ret = sdr.readStream(rx_stream, [buff], len(buff), timeoutUs=int(args.timeout_us))
            if sr_ret.ret > 0:
                n = sr_ret.ret
                iq = buff[:n]
                samples_total += n

                mdb, pdb = power_stats(iq)
                mean_db_acc.append(mdb)
                peak_db_acc.append(pdb)

                # Optional plotting
                if do_plot:
                    N = int(args.fft)
                    if n >= N:
                        x = iq[:N]
                    else:
                        # zero-pad if short read
                        x = np.zeros(N, np.complex64)
                        x[:n] = iq

                    spec = np.fft.fftshift(np.fft.fft(x, n=N))
                    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
                    line.set_ydata(mag_db)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            else:
                # ret <= 0 indicates error/timeout
                if sr_ret.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                    timeouts += 1
                elif sr_ret.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                    overruns += 1
                else:
                    other_errs += 1

            # Periodic report
            if (now - last_report) >= args.report_s:
                elapsed = now - start
                rate_eff = (samples_total / elapsed) if elapsed > 0 else 0.0

                if mean_db_acc:
                    m = float(np.mean(mean_db_acc[-10:]))
                    p = float(np.max(peak_db_acc[-10:]))
                else:
                    m, p = float("nan"), float("nan")

                print(f"t={elapsed:6.1f}s  eff_rate={rate_eff/1e6:6.3f} Msps  "
                      f"mean={m:7.2f} dB  peak={p:7.2f} dB  "
                      f"overruns={overruns}  timeouts={timeouts}  other={other_errs}")

                last_report = now

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

    elapsed = max(1e-9, time.time() - start)
    print("\n[summary]")
    print(f"  Duration:    {elapsed:.2f} s")
    print(f"  Samples:     {samples_total}")
    print(f"  Eff rate:    {samples_total/elapsed/1e6:.3f} Msps")
    if mean_db_acc:
        print(f"  Mean power:  {float(np.mean(mean_db_acc)):.2f} dB (avg over reads)")
        print(f"  Peak power:  {float(np.max(peak_db_acc)):.2f} dB (max over reads)")
    print(f"  Overruns:    {overruns}")
    print(f"  Timeouts:    {timeouts}")
    print(f"  Other errs:  {other_errs}")


def parse_args():
    p = argparse.ArgumentParser(description="SoapySDR bladeRF RX test (no GPS/compass).")
    p.add_argument("--driver", default="bladerf", help="Soapy driver to select (default: bladerf)")
    p.add_argument("--serial", default="", help="Optional SDR serial to select")
    p.add_argument("--index", type=int, default=None, help="Device index from enumerate() to open")

    p.add_argument("--freq", type=float, default=100e6, help="Center frequency (Hz)")
    p.add_argument("--rate", type=float, default=2e6, help="Sample rate (S/s). Start low on Jetson.")
    p.add_argument("--bw", type=float, default=None, help="Analog bandwidth (Hz) (optional)")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB (overall, if supported)")
    p.add_argument("--seconds", type=float, default=10.0, help="Run time (s)")
    p.add_argument("--timeout-us", type=int, default=200000, help="readStream timeout (microseconds)")
    p.add_argument("--report-s", type=float, default=1.0, help="Status print interval (s)")

    p.add_argument("--buf", type=int, default=0, help="RX buffer length in samples (0=auto)")
    p.add_argument("--plot", action="store_true", help="Show live spectrum plot")
    p.add_argument("--fft", type=int, default=4096, help="FFT size for --plot")

    return p.parse_args()


def main():
    args = parse_args()

    print("=== SDR Live Test (SoapySDR) ===")
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
    print("")

    print("[1] Enumerating and opening device...")
    picked = pick_device(args)
    print(f"[1] Opening: {picked}")
    sdr = SoapySDR.Device(picked)

    print("\n[2] Gain capabilities:")
    list_gain_elements(sdr)

    cf, sr = set_rx_params(sdr, args)
    run_stream_test(sdr, args, cf, sr)


if __name__ == "__main__":
    main()
