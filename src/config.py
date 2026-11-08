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

# Λονδίνο: IDs που επαναχρησιμοποιήθηκαν για άλλον σταθμό μετά το 2025.
# Η θέση του BikePoint δεν αντιστοιχεί στις διαδρομές του 2025.
#   001058: διαδρομές "Edgware Road Station, Marylebone", BikePoint "Orchard Place, Blackwall Tunnel"
LONDON_INVALID_STATION_IDS = ["001058"]

# ---- Split Λονδίνου (§2.4) ----
TEST_EVERY_NTH_DAY = 5          # κάθε 5η ημέρα του έτους στο test
VAL_EVERY_NTH_DAY = 10          # από τις μη-test ημέρες, κάθε 10η στο validation
LONDON_TEST_ROWS = 200_000
LONDON_TRAINVAL_ROWS = 800_000

# ---- OSM: προφίλ και τύποι σταθμών (§3) ----
OSM_DIR = CONTEXT_DIR / "osm"
PBF_DIR = RAW_DIR / "osm"         # τοπικά αρχεία OSM του Geofabrik (.osm.pbf)
CITY_CENTERS = {                  # κέντρο κάθε πόλης για την απόσταση από το κέντρο
    "chicago": (41.8819, -87.6278),   # The Loop
    "nyc":     (40.7549, -73.9840),   # Midtown Manhattan
    "london":  (51.5073, -0.1276),    # Charing Cross
}
PROFILE_RADIUS_M = 500            # ακτίνα των χαρακτηριστικών γύρω από κάθε σταθμό
OSM_BBOX_PAD_DEG = 0.01           # περιθώριο ~1 km γύρω από τους σταθμούς για τη λήψη OSM

# ---- Τύποι σταθμών (§3.2, 05b) ----
STATION_TYPE_K_RANGE = (5, 8)     # υποψήφια k για το k-means· επιλογή με silhouette
MIN_TYPE_SHARE = 0.02             # ελάχιστο μερίδιο αφετηριών (μείγμα 50/50 Chicago-NYC)
MIN_TRIPS_FOR_FIT = 10            # σταθμοί με λιγότερες διαδρομές δεν ορίζουν τύπους
