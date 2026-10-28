"""Download exactly the fixed validation subset and prepare aligned bbox truth."""
import concurrent.futures as cf
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import requests
from .core import read_image, resize_image, write_image

ANNOTATION_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
WEIGHT_MIRROR = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt"
WEIGHT_API = "https://api.github.com/repos/ultralytics/assets/releases/assets/196201276?download=1"
WEIGHT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写完临时文件再替换，避免程序中断后留下半份JSON。
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class RangeReader(io.RawIOBase):
    """Read the small val member without transferring the 253 MB official ZIP."""
    def __init__(self, url):
        self.url, self.position = url, 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("Server does not support byte ranges")
        self.size = int(r.headers["Content-Range"].split("/")[-1])
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position
    def read(self, n=-1):
        n = min(self.size - self.position, self.size if n < 0 else n)
        if n <= 0:
            return b""
        start = self.position
        # zipfile可能跳读中央目录和压缩数据，每次请求都按当前位置取指定字节。
        r = requests.get(self.url, headers={"Range": f"bytes={start}-{start+n-1}"}, timeout=(15, 60))
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != n:
            raise RuntimeError("Invalid byte-range response")
        self.position += n
        return r.content

def download(url, path, image=False, headers=None):
    path = Path(path)
    if path.exists():
        if image:
            read_image(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=(15, 45))
            r.raise_for_status()
            tmp = path.with_name(path.stem + ".part" + path.suffix)
            tmp.write_bytes(r.content)
            if image:
                read_image(tmp)
            tmp.replace(path)
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

def prepare(config):
    root = Path(config["data_dir"])
    ann_path = root / "annotations/instances_val2017.json"
    if not ann_path.exists():
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RangeReader(ANNOTATION_URL)) as z:
            ann_path.write_bytes(z.read("annotations/instances_val2017.json"))
    original = json.loads(ann_path.read_text(encoding="utf-8"))
    # 抽样前固定排序，避免源标注的排列变化影响选中的图像。
    images = sorted(original["images"], key=lambda x: x["id"])
    ids = np.random.default_rng(config["sample_seed"]).choice(
        len(images), size=config["sample_count"], replace=False)
    selected = sorted([images[int(i)] for i in ids], key=lambda x: x["id"])
    selected_ids = {im["id"] for im in selected}
    selected_anns = [a for a in original["annotations"] if a["image_id"] in selected_ids]
    def get_image(im):
        url = f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{im["file_name"]}'
        path = root / "original" / im["file_name"]
        download(url, path, image=True)
        return im, path, url
    # 下载可并发；map仍按输入顺序返回，后续图像与标注顺序保持一致。
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(get_image, selected))
    processed_images, transforms, manifest = [], {}, []
    for im, source, url in fetched:
        clean, sx, sy = resize_image(read_image(source), config["max_side"])
        target = root / "clean" / (Path(im["file_name"]).stem + ".png")
        write_image(target, clean)
        h, w = clean.shape[:2]
        transforms[im["id"]] = (sx, sy)
        processed_images.append({**im, "file_name": target.name, "width": w, "height": h})
        manifest.append({
            "image_id": im["id"], "original_file": im["file_name"], "clean_file": target.name,
            "source_url": url, "coco_url": im.get("coco_url"), "license": im.get("license"),
            "original_sha256": sha256(source), "clean_sha256": sha256(target),
            "original_size": [im["width"], im["height"]], "processed_size": [w, h],
            "scale_x": sx, "scale_y": sy,
            "noncrowd_objects": sum(a["image_id"] == im["id"] and not a.get("iscrowd", 0) for a in selected_anns),
        })
    anns = []
    for a in selected_anns:
        # 宽高取整后，横纵缩放比例可能略有差异；框与面积分别同步缩放。
        sx, sy = transforms[a["image_id"]]
        x, y, w, h = a["bbox"]
        anns.append({k: v for k, v in a.items() if k not in ["segmentation", "bbox", "area"]} |
                    {"bbox": [x*sx, y*sy, w*sx, h*sy], "area": a["area"]*sx*sy})
    truth = {
        "info": original.get("info", {}), "licenses": original.get("licenses", []),
        "images": processed_images, "annotations": anns, "categories": original["categories"],
    }
    save_json(root / "annotations/subset_resized.json", truth)
    save_json(root / "manifest.json", {"split": "COCO val2017", "seed": config["sample_seed"],
              "annotation_url": ANNOTATION_URL, "annotation_sha256": sha256(ann_path),
              "licenses": original.get("licenses", []), "images": manifest})
    if not Path(config["model"]).exists():
        try:
            download(WEIGHT_URL, config["model"])
        except requests.RequestException:
            try:
                print("Release URL failed; trying the official GitHub asset API.", flush=True)
                download(WEIGHT_API, config["model"], headers={"Accept": "application/octet-stream"})
            except requests.RequestException:
                download(WEIGHT_MIRROR, config["model"])
    # 所有下载来源都必须通过同一权重校验，防止模型版本悄然变化。
    if sha256(config["model"]) != WEIGHT_SHA256:
        raise ValueError("Unexpected model checksum; use the fixed official YOLO11n v8.3.0 asset")
    print(f"Prepared {len(selected)} images, {len(anns)} annotations; weights {sha256(config['model'])}", flush=True)
    return truth
