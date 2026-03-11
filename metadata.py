import json
import os

def extract_capture_metadata(json_path):
    """
    Parses IQ metadata, compass data, and GPS coordinates from a capture JSON file.
    Returns a dictionary of relevant parameters.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extracting core IQ parameters (using 'actual' values for accuracy)
    iq_params = {
        "filename": data.get("file_cf32"),
        "sample_rate": data.get("actual", {}).get("rate_sps"),
        "center_freq": data.get("actual", {}).get("freq_hz"),
        "gain_db": data.get("actual", {}).get("gain_db"),
        "dtype": data.get("dtype", "complex64")
    }

    # Extracting Sensor Data
    sensors = data.get("sensors", {})
    
    # Compass Info
    compass = sensors.get("compass", {})
    compass_data = {
        "heading_deg": compass.get("last_heading_deg"),
        "heading_csv": compass.get("heading_csv")
    }

    # GPS Info (Handles null values)
    gps = sensors.get("gps", {})
    gps_data = {
        "lat": gps.get("lat"),
        "lon": gps.get("lon"),
        "alt": gps.get("alt_m")
    }

    return {
        "iq": iq_params,
        "compass": compass_data,
        "gps": gps_data,
        "timestamp": data.get("timing", {}).get("start_utc")
    }

# Example Usage:
# metadata = extract_capture_metadata('cap_4_ethan.json')
# print(f"Processing {metadata['iq']['filename']} at {metadata['iq']['center_freq']} Hz")