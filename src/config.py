from pathlib import Path
# ---- Φάκελοι ----
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CONTEXT_DIR = Path("data/context")
TMP_DIR = Path("data/tmp")

# ---- Καθαρισμός (§2.2) — ίδια για όλες τις πόλεις, ορισμένα από Chicago/NYC ----
# (lat_min, lat_max, lng_min, lng_max): εκατοστημόρια 0,01–99,99%, στρογγυλεμένα προς τα έξω
CITY_BBOX = {
    "chicago": (41.60, 42.10, -87.90, -87.50),
    "nyc":     (40.60, 40.92, -74.10, -73.80),
}
MIN_TRIP_DURATION_MIN = 1.0      # αποκλειστικό
MAX_TRIP_DURATION_MIN = 100.0    # αποκλειστικό· >99ο εκατοστημόριο και στις δύο πόλεις
MIN_MOVING_DISTANCE_KM = 0.05    # μόνο για μη κυκλικές
MAX_TRIP_DISTANCE_KM = 12.0      # >99ο εκατοστημόριο
MAX_SPEED_KMH = 40.0             # μόνο για μη κυκλικές

# ---- Σταθμοί (§3) ----
MERGE_RADIUS_M = 25              # IDs σε απόσταση έως τόσο = ίδιος φυσικός σταθμός
MIN_CLASSIC_FOR_POSITION = 20    # θέση μόνο από classic αν υπάρχουν τουλάχιστον τόσες

# ---- Δειγματοληψία (§2.2) ----
TARGET_ROWS = 1_000_000
STRATA_COLS = ["month", "rideable_type", "part_of_day", "is_weekend"]
SAMPLING_SEED = 42