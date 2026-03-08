from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

PRODUCT_FILE = DATA_RAW / "product_info.csv"
REVIEW_FILES = [
    DATA_RAW / "reviews_0-250.csv",
    DATA_RAW / "reviews_250-500.csv",
    DATA_RAW / "reviews_500-750.csv",
    DATA_RAW / "reviews_750-1250.csv",
    DATA_RAW / "reviews_1250-end.csv",
]

RANDOM_STATE = 42
N_TOPICS = 5
MAX_TOPIC_SAMPLES = 50000