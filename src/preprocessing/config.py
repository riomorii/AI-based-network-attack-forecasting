from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Files
OUTPUT_FILE = PROCESSED_DATA_DIR / "cicids2018_processed.parquet"

# Dataset label column
LABEL_COLUMN = "Label"

# Features used by the first model
FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Avg Packet Size",
    "Down/Up Ratio",
]

# Final attack-family labels
LABEL_MAP = {
    "Benign": "BENIGN",

    "Bot": "BOT",

    "FTP-BruteForce": "BRUTE_FORCE",
    "SSH-Bruteforce": "BRUTE_FORCE",

    "DDoS attacks-LOIC-HTTP": "DDOS",
    "DDOS attack-HOIC": "DDOS",
    "DDOS attack-LOIC-UDP": "DDOS",

    "DoS attacks-GoldenEye": "DOS",
    "DoS attacks-Hulk": "DOS",
    "DoS attacks-Slowloris": "DOS",
    "DoS attacks-SlowHTTPTest": "DOS",

    "Infilteration": "INFILTRATION",

    "Brute Force -Web": "WEB_ATTACK",
    "Brute Force -XSS": "WEB_ATTACK",
    "SQL Injection": "WEB_ATTACK",
}

# Numeric model labels
LABEL_TO_ID = {
    "BENIGN": 0,
    "BOT": 1,
    "BRUTE_FORCE": 2,
    "DOS": 3,
    "DDOS": 4,
    "INFILTRATION": 5,
    "WEB_ATTACK": 6,
}

# Process data in chunks so the full 6.6M-flow dataset
# does not need to exist in RAM at once.
CHUNK_SIZE = 100_000