import argparse
import json
from pathlib import Path
from visionlab.experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run/resume the denoising experiment")
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--limit", type=int, help="Use first N manifest images for a smoke run")
    parser.add_argument("--condition-limit", type=int, help="Run only first N conditions")
    parser.add_argument("--output", help="Override output folder (keep practice separate)")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # 练习使用独立输出目录，正式记录保留在配置指定的位置。
    if args.output:
        config["output_dir"] = args.output
    run(config, args.limit, args.condition_limit)
