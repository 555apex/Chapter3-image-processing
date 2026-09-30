"""Run a paired matrix and resume only checkpoints whose inputs still match."""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import time
import numpy as np
from .core import Detector, add_noise, denoise, image_quality, read_image
from .data import save_json, sha256
from .evaluation import evaluate

def conditions(config):
    for method in config["methods"]:
        yield {"noise": "clean", "level": 0, "seed": 0, "method": method}
    for noise, levels in [("gaussian", config["gaussian_sigmas"]),
                           ("salt_pepper", config["salt_pepper_probabilities"])]:
        for level in levels:
            for seed in config["seeds"]:
                for method in config["methods"]:
                    yield {"noise": noise, "level": level, "seed": seed, "method": method}

def condition_key(c):
    return f'{c["noise"]}-{c["level"]}-s{c["seed"]}-{c["method"]}'

def load_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def run(config, limit=None, condition_limit=None):
    root, output = Path(config["data_dir"]), Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    truth = json.loads((root / "annotations/subset_resized.json").read_text(encoding="utf-8"))
    images = truth["images"][:limit] if limit else truth["images"]
    ids = {im["id"] for im in images}
    truth = {**truth, "images": images, "annotations": [a for a in truth["annotations"] if a["image_id"] in ids]}
    detector = Detector(config, truth["categories"])
    import torch
    torch.set_num_threads(4)
    fingerprint_inputs = {
        "config": config, "image_ids": sorted(ids), "model_sha256": sha256(config["model"]),
        "device": detector.device,
        "source_sha256": {name: sha256(Path(__file__).parent / name)
                          for name in ["core.py", "data.py", "evaluation.py", "experiment.py"]},
        "clean_sha256": {im["file_name"]: sha256(root / "clean" / im["file_name"]) for im in images},
        "annotation_sha256": sha256(root / "annotations/subset_resized.json"),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_inputs, sort_keys=True).encode()).hexdigest()
    environment = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": detector.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fallback_reason": detector.device_fallback, "inference_precision": "float32",
        "timing": "synchronous one-image inference, including preprocessing, NMS, and transfer; excludes disk I/O",
        "versions": {p: importlib.metadata.version(p) for p in
                     ["ultralytics", "opencv-python", "numpy", "scikit-image", "pycocotools"]},
        "model_sha256": fingerprint_inputs["model_sha256"], "fingerprint": fingerprint,
    }
    save_json(output / "environment.json", environment)
    save_json(output / "config.json", config)
    save_json(output / "annotations_resized.json", truth)
    shutil.copy2(root / "manifest.json", output / "data_manifest.json")
    clean_images = {im["id"]: read_image(root / "clean" / im["file_name"]) for im in images}
    matrix = list(conditions(config))
    if condition_limit:
        matrix = matrix[:condition_limit]
    results = []
    started = time.perf_counter()
    for n, c in enumerate(matrix, 1):
        key = condition_key(c)
        checkpoint = output / "conditions" / (key + ".json")
        records_path = output / "records" / (key + ".json.gz")
        if checkpoint.exists() and records_path.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                results.append(cached)
                print(f"[{n}/{len(matrix)}] resume {key}", flush=True)
                continue
            raise RuntimeError(f"Checkpoint inputs changed: {key}. Use a new output directory.")
        rows, all_predictions = [], []
        for im in images:
            image_id = im["id"]
            clean = clean_images[image_id]
            noisy = add_noise(clean, c["noise"], c["level"], c["seed"], image_id)
            start = time.perf_counter()
            processed = denoise(noisy, c["method"], config["filters"])
            denoise_ms = (time.perf_counter() - start) * 1000
            quality = image_quality(clean, processed)
            predictions, detection_ms = detector.predict(processed, image_id)
            all_predictions.extend(predictions)
            rows.append({
                "image_id": image_id, **quality, "denoise_ms": denoise_ms,
                "detection_ms": detection_ms, "predictions": predictions,
                "processed_sha256": hashlib.sha256(processed.tobytes()).hexdigest(),
            })
        metrics = evaluate(truth, all_predictions, config["report_confidence"])
        image_counts = metrics.pop("per_image_counts")
        for row in rows:
            row.update(image_counts[str(row["image_id"])])
        psnrs = [r["psnr_db"] for r in rows if r["psnr_db"] is not None]
        summary = {
            **c, "key": key, "image_count": len(images), **metrics, "fingerprint": fingerprint,
            "psnr_db": float(np.mean(psnrs)) if psnrs else None,
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "denoise_ms": float(np.mean([r["denoise_ms"] for r in rows])),
            "detection_ms": float(np.mean([r["detection_ms"] for r in rows])),
            "denoise_median_ms": float(np.median([r["denoise_ms"] for r in rows])),
            "detection_median_ms": float(np.median([r["detection_ms"] for r in rows])),
            "records_file": str(records_path.relative_to(output)).replace("\\", "/"),
        }
        records_path.parent.mkdir(exist_ok=True)
        temp = records_path.with_suffix(".part")
        with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(rows, f, ensure_ascii=False, allow_nan=False)
        temp.replace(records_path)
        save_json(checkpoint, summary)
        results.append(summary)
        print(f'[{n}/{len(matrix)}] {key}: AP={summary["map50_95"]:.4f}, F1={summary["f1"]:.4f}, {time.perf_counter()-started:.0f}s elapsed', flush=True)
    fields = [k for k in results[0] if k not in ["fingerprint", "records_file"]]
    with (output / "condition_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    grouped = {}
    numeric = ["map50", "map50_95", "precision", "recall", "f1", "psnr_db", "ssim", "denoise_ms", "detection_ms"]
    for result in results:
        group = (result["noise"], result["level"], result["method"])
        grouped.setdefault(group, []).append(result)
    aggregates = []
    for (noise, level, method), values in grouped.items():
        row = {"noise": noise, "level": level, "method": method, "repetitions": len(values)}
        for metric in numeric:
            numbers = [v[metric] for v in values if v[metric] is not None]
            row[metric + "_mean"] = float(np.mean(numbers)) if numbers else None
            row[metric + "_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else (0.0 if numbers else None)
        aggregates.append(row)
    save_json(output / "aggregate_results.json", aggregates)
    with (output / "aggregate_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    save_json(output / "completion.json", {
        "conditions_complete": len(results), "conditions_expected": len(matrix),
        "image_inferences": len(results)*len(images), "fingerprint": fingerprint,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_this_invocation_seconds": time.perf_counter()-started,
        "full_matrix": limit is None and condition_limit is None and len(results) == 114,
    })
    return aggregates
