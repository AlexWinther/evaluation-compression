from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

FAIRVISION_PROTECTED_ATTRIBUTES = (
    "age",
    "gender",
    "race",
    "ethnicity",
    "language",
    "maritalstatus",
)
CLASS_TO_IDX = {
    "dr": {"non-vision threatening dr": 0, "vision threatening dr": 1},
    "glaucoma": {"no": 0, "yes": 1},
    "amd": {"normal": 0, "early amd": 1, "intermediate amd": 2, "late amd": 3},
}
MODELS_DIR = PROJ_ROOT / "models"
MLFLOW_DB_PATH = PROJ_ROOT / "mlflow.db"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
