# SpectRe — Signal Processing Pipeline

SpectRe is a UAV-based beacon localization system. The drone flies a search pattern over a grid, rotating its directional antenna to collect IQ captures at known GPS positions. The offline processing pipeline converts those captures into line-of-bearing (LOB) estimates and triangulates a beacon location via heatmap and centroid localization.

---

## Repository Structure

```
├── main.py                  # Entry point — runs batch processing and localization
├── SpectRe_Processor.py     # Batch processor class + CONFIG
├── Signal_Downconverter.py  # DDC pipeline: baseband shift → filter → decimate
├── LoRa_Detector.py         # Matched filter, region extraction, LOB clustering
├── IQ_Visualizer.py         # IQ loading, metadata extraction, signal quality plots
├── Spectrogram.py           # Spectrogram viewer utility
├── geospatial.py            # SignalMapper: heatmap generation and centroid localization
```

---

## Pipeline Overview

```
Raw IQ Capture
     │
     ▼
Baseband Shift        — Frequency offset correction via complex mixing
     │
     ▼
Lowpass Filter        — Zero-phase FIR (filtfilt) to isolate beacon bandwidth
     │
     ▼
Decimation            — Sample rate reduction (600 kHz → 150 kHz)
     │
     ▼
Matched Filter        — FFT correlation against reference FMCW chirp
     │
     ▼
LOB Extraction        — Cluster detections, extract heading arc and centroid
     │
     ▼
Heading Association   — Interpolate compass to RF event timestamp
     │
     ▼
Heatmap + Centroid    — Combine all bearings into a spatial likelihood map
```

---

## Quickstart

### 1. Set up your capture folder

Captures should follow the naming convention defined in CONFIG:

```
Oval/
├── spin_capture_0.json     # Metadata: GPS, sample rate, center freq, compass offset
├── spin_capture_0.bin      # Raw IQ samples (cs16 format)
├── spin_capture_0.csv      # Compass headings (timestamped)
├── spin_capture_1.json
...
```

The metadata JSON is the source of truth for hardware parameters. Each file must contain:

| Field | Description |
|---|---|
| `IQ_file` | Filename of the raw IQ binary |
| `heading_file` | Filename of the compass CSV |
| `lat`, `lon` | GPS coordinates of the capture location |
| `freq` | SDR center frequency (Hz) |
| `rate` | SDR sample rate (Hz) |
| `log_delta_ns` | Hardware timing offset between IQ and compass (nanoseconds) |
| `heading_final` | Fallback heading if no compass CSV is present |

### 2. Configure `SpectRe_Processor.py`

All parameters live in the `CONFIG` dict at the top of `SpectRe_Processor.py`:

```python
CONFIG = {
    "folder_path":        "Oval/",         # Path to capture folder
    "file_prefix":        "spin_capture_", # Filename prefix before the index number
    "num_files":          21,              # Number of capture files to process
    "bearing_offset_deg": 0.0,             # Calibration offset: antenna boresight vs compass (degrees)
    "iq_format":          "cs16",          # IQ sample format (cs16 = complex int16)
    "target_freq":        915.5e6,         # Beacon frequency (Hz)
    "lora_sf":            12,              # LoRa spreading factor
    "lora_bw":            125e3,           # Beacon bandwidth (Hz)
    "decimation":         4,               # Decimation factor (600 kHz / 4 = 150 kHz)
    "threshold":          0.8,             # Matched filter detection threshold (0–1, normalized)
    "window_size":        1,               # Moving average window for correlation smoothing
    "tests_per_file":     1,               # Number of rotation segments per capture file
    "debug_plots":        False            # Set True to enable full stage-by-stage debug plots
}
```

### 3. Set the true beacon location (optional)

In `main.py`, uncomment the `true_loc` entry matching your test session. This is used only for validation overlay on the heatmap — it does not influence the localization estimate.

```python
# --- Oval ---
true_loc = (40.576238, -105.081204)

# --- IM ---
# true_loc = (40.573389, -105.090778)

# --- Apr 20 ---
# true_loc = (40.57334, -105.08939)
# ... etc.
```

### 4. Run

```bash
python main.py
```

Output:

```
--- SPECTRE REPORT ---
Estimated GPS: (lat, lon)
Confidence Box: 12.3m x 8.7m
Search Area: 107.0 m²
```

---

## Debug Mode

Setting `"debug_plots": True` in CONFIG enables a full visual trace of the pipeline for every capture segment. Plots fire stage by stage and block until closed.

### DDC stages (`Signal_Downconverter.py`)

| Stage | Plot |
|---|---|
| Stage 0 | Raw IQ — spectrum, time series, constellation (pre-shift) |
| Stage 1 | After baseband shift — spectrum centered at DC |
| Stage 2 | After lowpass filter — out-of-band energy removed |
| Stage 3 | After decimation — reduced sample rate, beacon bandwidth preserved |

Each stage renders three panels: power spectrum, I/Q time series, and constellation.

### Matched filter stages (`LoRa_Detector.py`)

| Stage | Plot |
|---|---|
| Stage 1 | Reference upchirp — spectrogram, I/Q, constellation |
| Stage 2 | Raw FFT convolution output with alignment boundaries marked |
| Stage 3 | Aligned correlation magnitude (pre-normalization) |
| Stage 4 | Normalized correlation with detection threshold |
| Stage 5 | Smoothed correlation (post moving average) |
| Stage 6 | Detected regions shaded over final correlation output |
| LOB     | Cluster size bar chart + detection timeline with centroid and LOB span |

---

## Key Parameters

### `threshold` (default: `0.8`)

Controls the detection sensitivity of the matched filter. After normalization, the correlation output is scaled to [0, 1]. Any sample above `threshold` is considered a detection. Lower values increase sensitivity but may produce false positives in noisy environments. Start at `0.8` and reduce if no detections are found.

### `bearing_offset_deg` (default: `0.0`)

Fixed angular offset between the antenna boresight and the compass sensor axis on the physical payload. This must be measured by calibration — pointing the boresight at a known target and recording the compass reading. An incorrect offset shifts every bearing estimate by the same angle, rotating the entire heatmap.

### `decimation` (default: `4`)

Reduces sample rate from the SDR capture rate to the processing rate. Must satisfy Nyquist for the beacon bandwidth: `fs_decimated > beacon_bw`. With complex (IQ) sampling, the effective two-sided bandwidth is doubled, so `150 kHz complex > 125 kHz beacon BW` is valid. Always filter before decimating.

### `tests_per_file` (default: `1`)

Number of rotation segments to split each capture into. Set to `3` if the drone completed three full rotations per capture file. Each segment is processed independently and contributes its own LOB estimate.

### `window_size` (default: `1`)

Moving average window applied to the correlation output before region extraction. Larger values smooth out noise spikes at the cost of temporal resolution on peak edges. Keep at `1` unless false detections are a problem.

---

## Output

`processor.process()` returns a list of dicts, one per detected LOB:

```python
{
    "source":         "spin_capture_0.json",  # Source file
    "test_id":        "0_0",                  # file_num_test_idx
    "lat":            40.576238,              # Receiver latitude
    "lon":           -105.081204,             # Receiver longitude
    "heading_start":  112.4,                  # Start of detected bearing arc (degrees)
    "heading_end":    138.7,                  # End of detected bearing arc (degrees)
    "heading_center": 125.1,                  # Peak/centroid heading (degrees)
    "max_corr":       0.94,                   # Peak normalized correlation value
    "weight":         1                       # Detection weight for heatmap (reserved)
}
```

These are passed directly to `SignalMapper.add_detections()` for localization.

---

## Dependencies

```
numpy
scipy
matplotlib
```

Internal modules: `IQ_Visualizer`, `Spectrogram`, `geospatial` (project-specific).