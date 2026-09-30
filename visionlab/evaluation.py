"""COCO evaluation and fixed-threshold counts with the same crowd-ignore rules."""
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(truth, predictions, confidence=0.25, image_ids=None):
    ids = image_ids or [im["id"] for im in truth["images"]]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = truth
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {"images": truth["images"], "categories": truth["categories"], "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    tp = fp = fn = 0
    per_image = {str(i): {"tp": 0, "fp": 0, "fn": 0} for i in ids}
    # COCO's IoU=.5 greedy matching and ignore flags; filtering the score-sorted
    # low-confidence tail cannot change matches of the retained detections.
    for item in ev.evalImgs:
        if item is None or item["aRng"] != ev.params.areaRng[0]:
            continue
        keep = np.asarray(item["dtScores"]) >= confidence
        matches = item["dtMatches"][0]
        ignored = item["dtIgnore"][0]
        n_tp = int(np.count_nonzero(keep & (matches > 0) & ~ignored))
        n_fp = int(np.count_nonzero(keep & (matches == 0) & ~ignored))
        matched_gt = set(matches[keep & ~ignored & (matches > 0)].astype(int))
        n_fn = sum(not ignore and int(gid) not in matched_gt
                   for gid, ignore in zip(item["gtIds"], item["gtIgnore"]))
        tp, fp, fn = tp+n_tp, fp+n_fp, fn+n_fn
        counts = per_image[str(item["image_id"])]
        counts["tp"] += n_tp
        counts["fp"] += n_fp
        counts["fn"] += n_fn
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "map50_95": float(ev.stats[0]), "map50": float(ev.stats[1]),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "per_image_counts": per_image,
        "categories_with_ground_truth": sum(
            any(a["category_id"] == c["id"] and a["image_id"] in ids and not a.get("iscrowd", 0)
                for a in truth["annotations"]) for c in truth["categories"]),
    }
