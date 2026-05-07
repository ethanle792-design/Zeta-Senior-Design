#!/usr/bin/env python3
"""
sdr_live_rx2_waterfall.py

Live spectrum + waterfall viewer for the second RX channel on a bladeRF via SoapySDR.
Built from the same style as the existing sdr_live.py, but adds:
- selectable RX channel (default: 1 for the second receiver)
- live spectrum plot
- rolling waterfall display

Examples:
  python3 sdr_live_rx2_waterfall.py --freq 2.437e9 --rate 5e6 --gain 30
  python3 sdr_live_rx2_waterfall.py --channel 1 --freq 2.462e9 --rate 10e6 --bw 8e6 --gain 35
  python3 sdr_live_rx2_waterfall.py --index 0 --channel 1 --freq 2.44e9 --rate 5e6 --seconds 0

Notes:
- --channel defaults to 1 because you said the 2.4 GHz antenna is on the second receiver.
- --seconds 0 means run until Ctrl+C.
- If the displayed level is too flat/noisy, try adjusting --gain and --rate.
"""

import argparse
import signal
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


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
        raise RuntimeError("No SoapySDR devices found. Try: SoapySDRUtil --find")

    if args.index is not None:
        if args.index < 0 or args.index >= len(devs):
            raise RuntimeError(f"--index {args.index} out of range (0..{len(devs)-1})")
        print(f"[pick] Using --index {args.index}")
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
        raise RuntimeError(f"Requested device not found (driver={args.driver!r}, serial={args.serial!r})")

    for d in devs:
        dd = kw_to_dict(d)
        if str(dd.get("driver", "")).lower() == "bladerf":
            return d

    print("[pick] No driver=bladerf found; falling back to device[0].")
    return devs[0]


def list_gain_elements(sdr, ch):
    try:
        gains = sdr.listGains(SOAPY_SDR_RX, ch)
        if gains:
            print(f"[gain] Channel {ch} elements: {gains}")
            for g in gains:
                try:
                    r = sdr.getGainRange(SOAPY_SDR_RX, ch, g)
                    print(f"       {g}: range {r.minimum()}..{r.maximum()} step={r.step()}")
                except Exception:
                    pass
        else:
            print(f"[gain] No gain elements reported for channel {ch}.")
    except Exception:
        print(f"[gain] listGains() not supported on channel {ch}.")


def set_rx_params(sdr, args):
    ch = args.channel
    sr = float(args.rate)
    cf = float(args.freq)

    sdr.setSampleRate(SOAPY_SDR_RX, ch, sr)
    try:
        actual_sr = sdr.getSampleRate(SOAPY_SDR_RX, ch)
    except Exception:
        actual_sr = sr

    sdr.setFrequency(SOAPY_SDR_RX, ch, cf)
    try:
        actual_cf = sdr.getFrequency(SOAPY_SDR_RX, ch)
    except Exception:
        actual_cf = cf

    if args.bw is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, ch, float(args.bw))
        except Exception:
            print("[warn] setBandwidth() not supported or failed; continuing.")

    try:
        actual_bw = sdr.getBandwidth(SOAPY_SDR_RX, ch)
    except Exception:
        actual_bw = None

    if args.gain is not None:
        try:
            sdr.setGain(SOAPY_SDR_RX, ch, float(args.gain))
        except Exception:
            ok = False
            for name in ("LNA", "VGA1", "VGA2", "RF", "IF"):
                try:
                    sdr.setGain(SOAPY_SDR_RX, ch, name, float(args.gain))
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                print("[warn] setGain failed; continuing.")

    print("\n[rx] Config:")
    print(f"  RX channel:   {ch}")
    print(f"  Center freq:  {actual_cf/1e6:.6f} MHz")
    print(f"  Sample rate:  {actual_sr/1e6:.6f} Msps")
    if actual_bw is not None:
        print(f"  Bandwidth:    {actual_bw/1e6:.6f} MHz")
    if args.gain is not None:
        try:
            g = sdr.getGain(SOAPY_SDR_RX, ch)
            print(f"  Gain:         {g:.1f} dB (reported)")
        except Exception:
            print(f"  Gain:         {args.gain:.1f} dB (set)")
    print("")

    return actual_cf, actual_sr


def main():
    args = parse_args()

    print("=== SDR Live RX Spectrum + Waterfall ===")
    print(f"Driver:       {args.driver}")
    if args.serial:
        print(f"Serial:       {args.serial}")
    if args.index is not None:
        print(f"Index:        {args.index}")
    print(f"RX channel:   {args.channel}")
    print(f"Center freq:  {args.freq/1e6:.3f} MHz")
    print(f"Sample rate:  {args.rate/1e6:.3f} Msps")
    if args.bw is not None:
        print(f"Bandwidth:    {args.bw/1e6:.3f} MHz")
    if args.gain is not None:
        print(f"Gain:         {args.gain:.1f} dB")
    print(f"FFT size:     {args.fft}")
    print(f"Waterfall:    {args.wf_lines} lines")
    print(f"Seconds:      {args.seconds} (0 = until Ctrl+C)")
    print("")

    picked = pick_device(args)
    print(f"[open] Opening: {picked}")
    sdr = SoapySDR.Device(picked)

    list_gain_elements(sdr, args.channel)
    cf, sr = set_rx_params(sdr, args)

    import matplotlib.pyplot as plt

    # bladeRF/Soapy quirk:
    # RX1 may not open cleanly as a single-channel stream [1],
    # even when the device reports 2 Rx channels.
    # For channel 1, open a 2-channel stream [0,1] and display only ch1.

    use_dual_stream = (args.channel == 1)

    if use_dual_stream:
        print("[stream] Opening dual-channel RX stream [0,1] and displaying RX1")
        # Make sure both channels are configured
        sdr.setSampleRate(SOAPY_SDR_RX, 0, float(args.rate))
        sdr.setFrequency(SOAPY_SDR_RX, 0, float(args.freq))
        if args.bw is not None:
            try:
                sdr.setBandwidth(SOAPY_SDR_RX, 0, float(args.bw))
            except Exception:
                pass
        if args.gain is not None:
            try:
                sdr.setGain(SOAPY_SDR_RX, 0, float(args.gain))
            except Exception:
                pass

        rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
    else:
        print(f"[stream] Opening single-channel RX stream [{args.channel}]")
        rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [args.channel])

    sdr.activateStream(rx_stream)

    try:
        mtu = int(sdr.getStreamMTU(rx_stream))
    except Exception:
        mtu = 8192

    buf_len = int(args.buf) if args.buf else max(args.fft, min(max(mtu, args.fft), 16384))

    if use_dual_stream:
        buff0 = np.empty(buf_len, np.complex64)
        buff1 = np.empty(buf_len, np.complex64)
    else:
        buff = np.empty(buf_len, np.complex64)
    try:
        mtu = int(sdr.getStreamMTU(rx_stream))
    except Exception:
        mtu = 8192

    buf_len = int(args.buf) if args.buf else max(args.fft, min(max(mtu, args.fft), 16384))
    buff = np.empty(buf_len, np.complex64)

    freqs = cf + np.fft.fftshift(np.fft.fftfreq(args.fft, d=1.0 / sr))
    window = np.hanning(args.fft).astype(np.float32)

    waterfall = np.full((args.wf_lines, args.fft), args.floor_db, dtype=np.float32)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.4]}
    )

    line, = ax1.plot(freqs / 1e6, np.full(args.fft, args.floor_db, dtype=float))
    ax1.set_title(f"Live Spectrum - RX{args.channel}")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
    ax1.set_ylim(-60, 20)
    ax1.grid(True)

    im = ax2.imshow(
        waterfall,
        aspect="auto",
        origin="lower",
        extent=[freqs[0] / 1e6, freqs[-1] / 1e6, 0, args.wf_lines],
        vmin=args.floor_db,
        vmax=args.ceil_db,
        interpolation="nearest",
    )
    ax2.set_title("Waterfall")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Time")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Magnitude (dB)")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    stop_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    start = time.time()
    last_report = start
    frames = 0
    timeouts = 0
    overruns = 0
    other_errs = 0

    print(f"[stream] MTU={mtu} using buf_len={buf_len}")
    print("[stream] Running... Ctrl+C to stop.\n")

    try:
        while True:
            now = time.time()
            if stop_flag["stop"]:
                break
            if args.seconds > 0 and (now - start) >= args.seconds:
                break

            if use_dual_stream:
                sr_ret = sdr.readStream(rx_stream, [buff0, buff1], len(buff0), timeoutUs=int(args.timeout_us))
            else:
                sr_ret = sdr.readStream(rx_stream, [buff], len(buff), timeoutUs=int(args.timeout_us))

            if sr_ret.ret > 0:
                n = sr_ret.ret

                if use_dual_stream:
                    # Display only RX1 since that's your 2.4 GHz path
                    iq = buff1[:n]
                else:
                    iq = buff[:n]

                if n >= args.fft:
                    x = iq[:args.fft]
                else:
                    x = np.zeros(args.fft, np.complex64)
                    x[:n] = iq

                xw = x * window
                spec = np.fft.fftshift(np.fft.fft(xw, n=args.fft))
                mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)

                line.set_ydata(mag_db)
                waterfall[:-1] = waterfall[1:]
                waterfall[-1] = mag_db.astype(np.float32)
                im.set_data(waterfall)

                fig.canvas.draw()
                fig.canvas.flush_events()
                frames += 1
            else:
                if sr_ret.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                    timeouts += 1
                elif sr_ret.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                    overruns += 1
                else:
                    other_errs += 1

            if (now - last_report) >= args.report_s:
                elapsed = now - start
                print(
                    f"t={elapsed:6.1f}s  frames={frames:6d}  "
                    f"timeouts={timeouts}  overruns={overruns}  other={other_errs}"
                )
                last_report = now

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

    elapsed = max(1e-9, time.time() - start)
    print("\n[summary]")
    print(f"  Duration:   {elapsed:.2f} s")
    print(f"  Frames:     {frames}")
    print(f"  Timeouts:   {timeouts}")
    print(f"  Overruns:   {overruns}")
    print(f"  Other errs: {other_errs}")


def parse_args():
    p = argparse.ArgumentParser(description="Live spectrum + waterfall on a selected bladeRF RX channel.")
    p.add_argument("--driver", default="bladerf", help="Soapy driver to select (default: bladerf)")
    p.add_argument("--serial", default="", help="Optional SDR serial to select")
    p.add_argument("--index", type=int, default=None, help="Device index from enumerate() to open")

    p.add_argument("--channel", type=int, default=1, help="RX channel index (default: 1 for second RX)")
    p.add_argument("--freq", type=float, default=2.437e9, help="Center frequency in Hz (default: 2.437e9)")
    p.add_argument("--rate", type=float, default=5e6, help="Sample rate in S/s")
    p.add_argument("--bw", type=float, default=None, help="Analog bandwidth in Hz")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB")

    p.add_argument("--seconds", type=float, default=0.0, help="Run time in seconds; 0 means until Ctrl+C")
    p.add_argument("--timeout-us", type=int, default=200000, help="readStream timeout in microseconds")
    p.add_argument("--report-s", type=float, default=1.0, help="Status print interval in seconds")
    p.add_argument("--buf", type=int, default=0, help="RX buffer length in samples (0=auto)")

    p.add_argument("--fft", type=int, default=2048, help="FFT size")
    p.add_argument("--wf-lines", type=int, default=200, help="Number of waterfall lines to retain")
    p.add_argument("--floor-db", type=float, default=-20.0, help="Display floor for spectrum/waterfall")
    p.add_argument("--ceil-db", type=float, default=90.0, help="Display ceiling for spectrum/waterfall")

    return p.parse_args()


if __name__ == "__main__":
    main()
