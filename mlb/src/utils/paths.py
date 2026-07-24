from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL):
    _p.mkdir(parents=True, exist_ok=True)
