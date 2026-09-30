"""Image operations shared by the demonstration and the full experiment."""
import hashlib
import os
import time
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity

cv2.setNumThreads(1)
DEFAULT_FILTERS = {
    "kernel": 3, "gaussian_sigma": 1, "bilateral_d": 5,
    "sigma_color": 25, "sigma_space": 25, "nlm_h": 10,
    "nlm_h_color": 10, "template_window": 7, "search_window": 21,
}
METHOD_NAMES = {
    "none": "不去噪", "mean": "均值", "gaussian": "高斯",
    "median": "中值", "bilateral": "双边", "nlm": "非局部均值",
}

def read_image(path):
    """imdecode handles Chinese/space-containing Windows paths."""
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image

def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buffer.tofile(path)

def resize_image(image, max_side=640):
    """No upsampling. Return exact independent x/y annotation scale factors."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    return resized, new_w / width, new_h / height

def add_noise(image, kind, level, seed, image_id):
    """Stable per-image random stream; seed never depends on method or Python hash."""
    key = f"{seed}|{image_id}|{kind}|{level}".encode("ascii")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    if kind == "clean":
        return image.copy()
    if kind == "gaussian":
        noise = rng.normal(0.0, float(level), image.shape)
        return np.rint(np.clip(image.astype(np.float32) + noise, 0, 255)).astype(np.uint8)
    if kind == "salt_pepper":
        if not 0 <= float(level) <= 1:
            raise ValueError("Salt-and-pepper probability must be in [0, 1]")
        draw = rng.random(image.shape[:2])
        noisy = image.copy()
        noisy[draw < level / 2] = 0
        noisy[(draw >= level / 2) & (draw < level)] = 255
        return noisy
    raise ValueError(f"Unknown noise: {kind}")

def denoise(image, method, parameters=None):
    p = {**DEFAULT_FILTERS, **(parameters or {})}
    kernel = int(p["kernel"])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Filter kernel must be a positive odd integer")
    if method == "none":
        return image.copy()
    if method == "mean":
        return cv2.blur(image, (kernel, kernel), borderType=cv2.BORDER_REFLECT_101)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), p["gaussian_sigma"], borderType=cv2.BORDER_REFLECT_101)
    if method == "median":
        return cv2.medianBlur(image, kernel)
    if method == "bilateral":
        return cv2.bilateralFilter(image, p["bilateral_d"], p["sigma_color"], p["sigma_space"], borderType=cv2.BORDER_REFLECT_101)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(image, None, p["nlm_h"], p["nlm_h_color"], p["template_window"], p["search_window"])
    raise ValueError(f"Unknown method: {method}")

def image_quality(clean, processed):
    """PSNR on all BGR channels, SSIM averaged over channels; both use range 255."""
    if clean.shape != processed.shape:
        raise ValueError("Quality inputs must have the same shape")
    mse = float(np.mean((clean.astype(np.float64) - processed.astype(np.float64)) ** 2))
    psnr = None if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    ssim = float(structural_similarity(clean, processed, data_range=255, channel_axis=2))
    return {"psnr_db": psnr, "ssim": ssim, "mse": mse}

class Detector:
    """Fixed detector. BGR arrays are passed directly to the Ultralytics API."""
    def __init__(self, config, categories):
        cache = Path(__file__).resolve().parents[1] / ".cache/ultralytics"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.config = config
        requested = config.get("device", "auto")
        self.device = ("0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        self.model = YOLO(config["model"])
        self.category_ids = [c["id"] for c in sorted(categories, key=lambda c: c["id"])]
        if len(self.category_ids) != 80:
            raise ValueError("YOLO COCO weights require the complete 80-category mapping")
        self.device_fallback = None
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            for _ in range(3):
                self.predict(dummy, -1)
        except (RuntimeError, NotImplementedError) as exc:
            if requested != "auto" or self.device == "cpu":
                raise
            self.device_fallback = str(exc)
            self.device = "cpu"
            for _ in range(3):
                self.predict(dummy, -1)

    def predict(self, image, image_id):
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.model.predict(
            source=image, imgsz=self.config["image_size"], device=self.device,
            conf=self.config["inference_confidence"], iou=self.config["nms_iou"],
            max_det=self.config["max_detections"], quantize=32, augment=False,
            verbose=False, rect=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        predictions = []
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            predictions.append({
                "image_id": int(image_id), "category_id": self.category_ids[cls],
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score),
            })
        return predictions, elapsed

def draw_boxes(image, annotations, predictions, confidence=0.25):
    canvas = image.copy()
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        cv2.rectangle(canvas, (round(x), round(y)), (round(x+w), round(y+h)), (0, 200, 0), 2)
    for pred in predictions:
        if pred["score"] < confidence:
            continue
        x, y, w, h = pred["bbox"]
        cv2.rectangle(canvas, (round(x), round(y)), (round(x+w), round(y+h)), (0, 0, 240), 2)
        cv2.putText(canvas, f'{pred["category_id"]}:{pred["score"]:.2f}',
                    (round(x), max(12, round(y)-3)), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 240), 1)
    return canvas
