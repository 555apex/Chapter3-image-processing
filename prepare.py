import argparse
import json
from pathlib import Path
from visionlab.data import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the fixed COCO validation subset and YOLO11n weights")
    parser.add_argument("--config", default="configs/full.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prepare(config)
