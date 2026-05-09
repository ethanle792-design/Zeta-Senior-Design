from SpectRe_Processor import SpectReProcessor, CONFIG
from geospatial import SignalMapper


def main():
    # ==========================================
    # --- BATCH PROCESSING ---
    # ==========================================
    # Instantiate the processor with the shared CONFIG and run the full pipeline.
    # Results are a list of LOB estimate dicts, one per detected beacon pass.
    processor = SpectReProcessor(CONFIG)
    results = processor.process()

    if not results:
        print("[!] No LOB estimates generated. Check capture files and threshold settings.")
        return

    # ==========================================
    # --- LOCALIZATION ---
    # ==========================================
    # Build the spatial likelihood map from all bearing estimates.
    # Resolution is in metres per grid cell; width/height define the map extent.
    mapper = SignalMapper(width=500, height=500, resolution=1)
    mapper.add_detections(results)

    # ==========================================
    # --- TRUE BEACON LOCATIONS (for validation) ---
    # ==========================================
    # Uncomment the relevant location before running.
    # These are used only for validation overlays — they do not influence the estimate.

    # --- Test 1 Apr 8 ---
    # true_loc = (40.59084, -105.14109)

    # --- Test 2 Apr 8 ---
    # true_loc = (40.59147, -105.14148)

    # --- Oval ---
    true_loc = (40.576238, -105.081204)

    # --- IM ---
    # true_loc = (40.573389, -105.090778)

    # --- IM2 ---
    # true_loc = (40.573389, -105.089333)

    # --- Apr 20 ---
    # true_loc = (40.57334, -105.08939)

    # --- Apr 1 Spin ---
    # true_loc = (40.576169, -105.080819)

    # ==========================================
    # --- RESULTS ---
    # ==========================================
    fix, bbox = mapper.plot(true_coords=true_loc)

    print(f"\n--- SPECTRE REPORT ---")
    print(f"Estimated GPS: {fix['coords_gps']}")
    if bbox:
        print(f"Confidence Box: {bbox['width']:.1f}m x {bbox['height']:.1f}m")
        print(f"Search Area:    {fix['search_area_m2']:.1f} m²")


if __name__ == "__main__":
    main()  