"""A staged entry point using exactly the same functions as the full experiment."""
import argparse
import json
from pathlib import Path
from visionlab.core import Detector, add_noise, denoise, draw_boxes, image_quality, read_image, write_image
from visionlab.data import save_json
from visionlab.evaluation import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--stage", choices=["read", "noise", "denoise", "detect"], default="detect")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--noise", choices=["gaussian", "salt_pepper"], default="gaussian")
    parser.add_argument("--level", type=float, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", type=int)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", default="practice-output/demo")
    args = parser.parse_args()
    # Match JSON's integer noise strengths exactly in the deterministic RNG key.
    # argparse's float 25.0 and the formal config's int 25 must share one stream.
    if float(args.level).is_integer():
        args.level = int(args.level)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    truth = json.loads((Path(config["data_dir"]) / "annotations/subset_resized.json").read_text(encoding="utf-8"))
    image = next((im for im in truth["images"] if im["id"] == args.image_id), None) if args.image_id else truth["images"][0]
    if image is None:
        raise ValueError("Image ID is not in the fixed subset")
    clean = read_image(Path(config["data_dir"]) / "clean" / image["file_name"])
    output = Path(args.output)
    write_image(output / "clean.png", clean)
    print(f'ID={image["id"]}, shape={clean.shape}, dtype={clean.dtype}, BGR range={clean.min()}..{clean.max()}')
    if args.stage == "read":
        return
    noisy = add_noise(clean, args.noise, args.level, args.seed, image["id"])
    write_image(output / "noisy.png", noisy)
    if args.stage == "noise":
        save_json(output / "quality.json", image_quality(clean, noisy))
        return
    if args.kernel:
        config["filters"]["kernel"] = args.kernel
    outputs = {method: denoise(noisy, method, config["filters"]) for method in config["methods"]}
    quality = {}
    for method, processed in outputs.items():
        write_image(output / (method + ".png"), processed)
        quality[method] = image_quality(clean, processed)
    save_json(output / "quality.json", quality)
    if args.stage == "denoise":
        return
    detector = Detector(config, truth["categories"])
    annotations = [a for a in truth["annotations"] if a["image_id"] == image["id"]]
    records = {}
    for method, processed in outputs.items():
        predictions, ms = detector.predict(processed, image["id"])
        result = evaluate(truth, predictions, args.confidence, [image["id"]])
        records[method] = {"predictions": predictions, "metrics": result, "detection_ms": ms}
        write_image(output / (method + "_boxes.png"), draw_boxes(processed, annotations, predictions, args.confidence))
        print(f'{method}: TP={result["tp"]} FP={result["fp"]} FN={result["fn"]}; {ms:.1f}ms')
    save_json(output / "detections.json", records)

if __name__ == "__main__":
    main()
