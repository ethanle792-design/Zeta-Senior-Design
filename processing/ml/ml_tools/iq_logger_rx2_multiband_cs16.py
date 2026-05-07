#!/usr/bin/env python3
"""
iq_logger_rx2_multiband_cs16.py

Capture CS16 IQ from the second RX path on a bladeRF via SoapySDR.
Designed for dataset collection / RadioML-style preprocessing.

Key features:
- Defaults to RX channel 1 (second receiver)
- Uses bladeRF dual-stream workaround for RX1: opens [0,1] and records only RX1
- Stores IQ as CS16 (interleaved int16 I/Q)
- Supports one or more predefined bands in a single run
- Default sample rate is 25 Msps
- Writes one .cs16 + one .json per band capture

Default band presets:
- zigbee    -> 2.425 GHz
- bluetooth -> 2.441 GHz
- wifi      -> 2.437 GHz

Examples:
  # Capture all three presets, 2 seconds each
  python3 iq_logger_rx2_multiband_cs16.py --duration 2

  # Capture only Wi-Fi
  python3 iq_logger_rx2_multiband_cs16.py --bands wifi --duration 3

  # Two rounds of all 3 bands
  python3 iq_logger_rx2_multiband_cs16.py --duration 1.5 --rounds 2

  # Override one preset frequency
  python3 iq_logger_rx2_multiband_cs16.py --duration 2 --wifi-freq 2.462e9

Notes:
- The default frequencies are just sensible starting points for collection.
- If you want exact channels you can override each one on the CLI.
- With 25 Msps, disk throughput and USB stability matter.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX


STOP = False


BAND_DEFAULTS_HZ: Dict[str, float] = {
    "zigbee": 2.425e9,
    "bluetooth": 2.441e9,
    "wifi": 2.437e9,
}


def _sigint_handler(signum, frame):
    global STOP
    STOP = True


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def kw_to_dict(k) -> dict:
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


def list_gain_elements(sdr, ch: int) -> None:
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


def set_gain_safe(sdr, ch: int, gain_db: float | None) -> None:
    if gain_db is None:
        return

    try:
        sdr.setGain(SOAPY_SDR_RX, ch, float(gain_db))
        return
    except Exception:
        pass

    for name in ("LNA", "VGA1", "VGA2", "RF", "IF"):
        try:
            sdr.setGain(SOAPY_SDR_RX, ch, name, float(gain_db))
            return
        except Exception:
            continue

    print(f"[warn] setGain failed on channel {ch}; continuing.")


def configure_channel(sdr, ch: int, freq_hz: float, rate_sps: float, bw_hz: float | None, gain_db: float | None) -> dict:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, float(rate_sps))
    sdr.setFrequency(SOAPY_SDR_RX, ch, float(freq_hz))

    if bw_hz is not None:
        try:
            sdr.setBandwidth(SOAPY_SDR_RX, ch, float(bw_hz))
        except Exception:
            print(f"[warn] setBandwidth failed on channel {ch}; continuing.")

    set_gain_safe(sdr, ch, gain_db)

    actual = {
        "freq_hz": float(freq_hz),
        "rate_sps": float(rate_sps),
        "bandwidth_hz": None if bw_hz is None else float(bw_hz),
        "gain_db": gain_db,
    }

    try:
        actual["freq_hz"] = float(sdr.getFrequency(SOAPY_SDR_RX, ch))
    except Exception:
        pass
    try:
        actual["rate_sps"] = float(sdr.getSampleRate(SOAPY_SDR_RX, ch))
    except Exception:
        pass
    try:
        actual["bandwidth_hz"] = float(sdr.getBandwidth(SOAPY_SDR_RX, ch))
    except Exception:
        pass
    try:
        actual["gain_db"] = float(sdr.getGain(SOAPY_SDR_RX, ch))
    except Exception:
        pass

    return actual


def parse_band_list(text: str) -> List[str]:
    bands = [b.strip().lower() for b in text.split(",") if b.strip()]
    valid = set(BAND_DEFAULTS_HZ.keys())
    bad = [b for b in bands if b not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown band(s): {bad}. Valid: {sorted(valid)}")
    if not bands:
        raise argparse.ArgumentTypeError("At least one band is required.")
    return bands


def parse_args():
    p = argparse.ArgumentParser(description="Capture RX2 IQ to CS16 across one or more preset 2.4 GHz bands.")

    p.add_argument("--driver", default="bladerf", help="Soapy driver to select (default: bladerf)")
    p.add_argument("--serial", default="", help="Optional SDR serial to select")
    p.add_argument("--index", type=int, default=None, help="Device index from enumerate() to open")

    p.add_argument("--channel", type=int, default=1, help="RX channel index (default: 1 for second RX)")
    p.add_argument("--rate", type=float, default=25e6, help="Sample rate in S/s (default: 25e6)")
    p.add_argument("--bw", type=float, default=None, help="Analog bandwidth in Hz")
    p.add_argument("--gain", type=float, default=None, help="Gain in dB")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--duration", type=float, help="Capture duration per band in seconds")
    mode.add_argument("--samples", type=int, help="Capture exactly N complex samples per band")

    p.add_argument("--bands", type=parse_band_list, default=["zigbee", "bluetooth", "wifi"],
                   help="Comma-separated preset list (default: zigbee,bluetooth,wifi)")
    p.add_argument("--rounds", type=int, default=1, help="How many times to cycle through the band list")

    p.add_argument("--zigbee-freq", type=float, default=BAND_DEFAULTS_HZ["zigbee"], help="Zigbee center frequency in Hz")
    p.add_argument("--bluetooth-freq", type=float, default=BAND_DEFAULTS_HZ["bluetooth"], help="Bluetooth center frequency in Hz")
    p.add_argument("--wifi-freq", type=float, default=BAND_DEFAULTS_HZ["wifi"], help="Wi-Fi center frequency in Hz")

    p.add_argument("--out", default="./captures", help="Output directory (default: ./captures)")
    p.add_argument("--name-prefix", default="rx2_capture", help="Base prefix for output files")
    p.add_argument("--label", default=None, help="Optional dataset label to write into metadata")

    p.add_argument("--block", type=int, default=262144, help="Read block size in complex samples")
    p.add_argument("--timeout-us", type=int, default=250000, help="readStream timeout in microseconds")
    p.add_argument("--settle-ms", type=int, default=150, help="Retune settle delay before each capture")

    return p.parse_args()


def open_stream_for_channel(sdr, ch: int):
    use_dual_stream = (ch == 1)
    if use_dual_stream:
        print("[stream] Opening dual-channel RX stream [0,1] and recording RX1")
        rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
    else:
        print(f"[stream] Opening single-channel RX stream [{ch}]")
        rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
    sdr.activateStream(rx_stream)
    return rx_stream, use_dual_stream


def make_capture_plan(args) -> List[Tuple[str, float]]:
    freq_map = {
        "zigbee": float(args.zigbee_freq),
        "bluetooth": float(args.bluetooth_freq),
        "wifi": float(args.wifi_freq),
    }
    plan: List[Tuple[str, float]] = []
    for r in range(args.rounds):
        for band in args.bands:
            plan.append((band, freq_map[band]))
    return plan


def capture_one_band(
    sdr,
    rx_stream,
    use_dual_stream: bool,
    args,
    band: str,
    freq_hz: float,
    capture_idx: int,
) -> dict:
    global STOP

    if args.channel == 1:
        configure_channel(sdr, 0, freq_hz, args.rate, args.bw, args.gain)
    actual = configure_channel(sdr, args.channel, freq_hz, args.rate, args.bw, args.gain)

    if args.settle_ms > 0:
        time.sleep(args.settle_ms / 1000.0)

    ts = utc_ts()
    base = f"{args.name_prefix}_{band}_{capture_idx:03d}_{ts}"
    cs16_path = os.path.join(args.out, base + ".cs16")
    json_path = os.path.join(args.out, base + ".json")

    if args.samples is not None:
        target_samples = int(args.samples)
        target_mode = "samples"
    else:
        target_samples = int(round(float(args.duration) * float(actual["rate_sps"])))
        target_mode = "duration"

    buf_len = int(args.block)
    if use_dual_stream:
        buff0 = np.empty(buf_len, np.complex64)
        buff1 = np.empty(buf_len, np.complex64)
    else:
        buff = np.empty(buf_len, np.complex64)

    written = 0
    timeouts = 0
    overruns = 0
    other_errs = 0
    first_block_mono_ns = None

    start_wall = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    print(f"\n[capture {capture_idx:03d}] {band} @ {actual['freq_hz']/1e6:.6f} MHz")
    print(f"  target: {target_samples} samples ({target_mode})")
    print(f"  file:   {cs16_path}")

    with open(cs16_path, "wb", buffering=0) as f:
        while not STOP and written < target_samples:
            need = min(buf_len, target_samples - written)

            if use_dual_stream:
                sr_ret = sdr.readStream(rx_stream, [buff0, buff1], need, timeoutUs=int(args.timeout_us))
            else:
                sr_ret = sdr.readStream(rx_stream, [buff], need, timeoutUs=int(args.timeout_us))

            if sr_ret.ret > 0:
                if first_block_mono_ns is None:
                    first_block_mono_ns = time.monotonic_ns()

                iq = buff1[:sr_ret.ret] if use_dual_stream else buff[:sr_ret.ret]

                i = np.clip(iq.real * 32767.0, -32768, 32767).astype(np.int16)
                q = np.clip(iq.imag * 32767.0, -32768, 32767).astype(np.int16)

                interleaved = np.empty(i.size * 2, dtype=np.int16)
                interleaved[0::2] = i
                interleaved[1::2] = q

                f.write(interleaved.tobytes())
                written += sr_ret.ret
                continue

            if sr_ret.ret == SoapySDR.SOAPY_SDR_TIMEOUT or sr_ret.ret == 0:
                timeouts += 1
            elif sr_ret.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                overruns += 1
            else:
                other_errs += 1
                time.sleep(0.001)

    end_wall = time.time()
    end_utc = datetime.now(timezone.utc).isoformat()

    meta = {
        "file": os.path.basename(cs16_path),
        "format": "cs16",
        "dtype": "interleaved int16 IQ",
        "label": args.label if args.label is not None else band,
        "band": band,
        "rx_channel": args.channel,
        "driver": args.driver,
        "requested": {
            "freq_hz": freq_hz,
            "rate_sps": args.rate,
            "bandwidth_hz": args.bw,
            "gain_db": args.gain,
            "duration_s": args.duration,
            "samples": args.samples,
            "block_samples": args.block,
            "timeout_us": args.timeout_us,
            "settle_ms": args.settle_ms,
        },
        "actual": actual,
        "timing": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "elapsed_s": end_wall - start_wall,
            "first_iq_block_monotonic_ns": first_block_mono_ns,
        },
        "stats": {
            "samples_written": written,
            "complex_samples_target": target_samples,
            "timeouts": timeouts,
            "overruns": overruns,
            "other_errs": other_errs,
        },
        "notes": [
            "IQ stored as interleaved int16: I0,Q0,I1,Q1,...",
            "For RX channel 1, the script opens a dual-channel stream [0,1] and records only RX1.",
            "This file is intended for later conversion to model tensors.",
        ],
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    print(f"  done: wrote {written} samples in {end_wall - start_wall:.3f} s")
    if timeouts or overruns or other_errs:
        print(f"  stats: timeouts={timeouts} overruns={overruns} other={other_errs}")

    return meta


def main():
    signal.signal(signal.SIGINT, _sigint_handler)
    args = parse_args()
    mkdir(args.out)

    if args.rounds < 1:
        raise RuntimeError("--rounds must be >= 1")

    print("=== RX2 Multiband CS16 Logger ===")
    print(f"Driver:      {args.driver}")
    if args.serial:
        print(f"Serial:      {args.serial}")
    if args.index is not None:
        print(f"Index:       {args.index}")
    print(f"RX channel:  {args.channel}")
    print(f"Rate:        {args.rate/1e6:.3f} Msps")
    if args.bw is not None:
        print(f"Bandwidth:   {args.bw/1e6:.3f} MHz")
    if args.gain is not None:
        print(f"Gain:        {args.gain:.1f} dB")
    print(f"Bands:       {args.bands}")
    print(f"Rounds:      {args.rounds}")
    print(f"Output dir:  {args.out}")
    print("")

    picked = pick_device(args)
    print(f"[open] Opening: {picked}")
    sdr = SoapySDR.Device(picked)

    list_gain_elements(sdr, args.channel)
    if args.channel == 1:
        list_gain_elements(sdr, 0)

    rx_stream, use_dual_stream = open_stream_for_channel(sdr, args.channel)

    try:
        plan = make_capture_plan(args)
        summaries = []
        for idx, (band, freq_hz) in enumerate(plan, start=1):
            if STOP:
                break
            summaries.append(capture_one_band(sdr, rx_stream, use_dual_stream, args, band, freq_hz, idx))

        print("\n[summary]")
        print(f"  Planned captures:  {len(plan)}")
        print(f"  Completed captures:{len(summaries)}")
        for item in summaries:
            print(f"  - {item['band']:9s} {item['actual']['freq_hz']/1e6:10.6f} MHz  {item['stats']['samples_written']} samples  {item['file']}")

    finally:
        try:
            sdr.deactivateStream(rx_stream)
        except Exception:
            pass
        try:
            sdr.closeStream(rx_stream)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
