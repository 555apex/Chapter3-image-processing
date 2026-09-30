"""Meaningful checks for paired noise, coordinate transforms, and COCO counting."""
import unittest
import numpy as np
from visionlab.core import add_noise, denoise, image_quality, resize_image
from visionlab.evaluation import evaluate

class Verification(unittest.TestCase):
    def test_reproducible_noise_and_input_immutability(self):
        clean = np.full((20, 30, 3), 100, np.uint8)
        first = add_noise(clean, "gaussian", 25, 42, 7)
        self.assertTrue(np.array_equal(first, add_noise(clean, "gaussian", 25, 42, 7)))
        self.assertFalse(np.array_equal(first, add_noise(clean, "gaussian", 25, 43, 7)))
        for method in ["none", "mean", "gaussian", "median", "bilateral", "nlm"]:
            saved = first.copy()
            output = denoise(first, method)
            self.assertTrue(np.array_equal(first, saved))
            self.assertEqual(output.shape, clean.shape)
            self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.all(clean == 100))

    def test_resize_and_quality(self):
        clean = np.full((1001, 701, 3), 100, np.uint8)
        resized, sx, sy = resize_image(clean, 640)
        self.assertEqual(resized.shape[:2], (640, 448))
        self.assertAlmostEqual(sx * 701, 448)
        self.assertAlmostEqual(sy * 1001, 640)
        quality = image_quality(resized, resized)
        self.assertIsNone(quality["psnr_db"])  # JSON null denotes +infinity, not zero.
        self.assertEqual(quality["ssim"], 1.0)
        with self.assertRaises(ValueError):
            denoise(resized, "median", {"kernel": 4})

    def test_coco_duplicates_classes_crowds_and_empty_predictions(self):
        truth = {
            "images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 1},
            ],
        }
        pred = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .8},
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": .7},
            {"image_id": 2, "category_id": 1, "bbox": [55, 55, 5, 5], "score": .9},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 10, 10], "score": .1},
        ]
        result = evaluate(truth, pred)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 0))
        self.assertEqual(result["precision"], 1/3)
        result = evaluate(truth, [])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))

if __name__ == "__main__":
    unittest.main(verbosity=2)
