from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# TRAINING SOURCE FILES
# ---------------------------------------------------------
# The scaler is fitted ONLY on these files.
TRAIN_FILES = [
    "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet",
    "Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",
]


# ---------------------------------------------------------
# VALIDATION SOURCE FILES
# ---------------------------------------------------------
# These files are NEVER used to fit the scaler.
VALIDATION_FILES = [
    "DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet",
]


# ---------------------------------------------------------
# FINAL UNSEEN TEST SOURCE FILES
# ---------------------------------------------------------
# These remain untouched until final evaluation.
TEST_FILES = [
    "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet",
]


def get_train_paths():
    return [RAW_DATA_DIR / filename for filename in TRAIN_FILES]


def get_validation_paths():
    return [RAW_DATA_DIR / filename for filename in VALIDATION_FILES]


def get_test_paths():
    return [RAW_DATA_DIR / filename for filename in TEST_FILES]