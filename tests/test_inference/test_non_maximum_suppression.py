import unittest
from shapely.geometry import Polygon
import numpy as np
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import non_maximum_suppression

class TestNonMaximumSuppression(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
    
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_suppression_no_touching(self):
        # Test case 1: No suppression
        contours = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[2, 2], [2, 3], [3, 3], [3, 2]]
        ]
        scores = [0.9, 0.8]
        height = 4
        width = 4
        threshold = 0.3
        filtered_contours, filtered_scores = non_maximum_suppression(
            contours, scores, height, width, threshold)
        self.assertEqual(len(filtered_contours), 2)
        self.assertEqual(len(filtered_scores), 2)

    def test_suppression_with_suppression(self):
        # Test case 2: Suppression occurs
        contours = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[0, 0], [0, 2], [2, 2], [2, 0]],
            [[2, 2], [2, 3], [3, 3], [3, 2]]
        ]
        scores = [0.8, 0.9, 0.7]
        height = 4
        width = 4
        threshold = 0.2
        filtered_contours, filtered_scores = non_maximum_suppression(
            contours, scores, height, width, threshold)
        self.assertEqual(len(filtered_contours), 2)
        self.assertEqual(len(filtered_scores), 2)

    def test_suppression_with_suppression_not_quiet(self):
        # Test case 2: Suppression occurs
        contours = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[0, 0], [0, 2], [2, 2], [2, 0]],
            [[2, 2], [2, 3], [3, 3], [3, 2]]
        ]
        scores = [0.8, 0.9, 0.7]
        height = 4
        width = 4
        threshold = 0.2
        filtered_contours, filtered_scores = non_maximum_suppression(
            contours, scores, height, width, threshold, quiet=False)
        self.assertEqual(len(filtered_contours), 2)
        self.assertEqual(len(filtered_scores), 2)

    def test_suppression_with_output(self):
        # Test case 2: Suppression occurs
        contours = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[0, 0], [0, 2], [2, 2], [2, 0]],
            [[2, 2], [2, 3], [3, 3], [3, 2]]
        ]
        scores = [0.8, 0.9, 0.7]
        height = 4
        width = 4
        threshold = 0.2
        filtered_contours, filtered_scores = non_maximum_suppression(
            contours, scores, height, width, threshold, output_name=f"{self.tmp_dir}/nms.png")
        self.assertEqual(len(filtered_contours), 2)
        self.assertEqual(len(filtered_scores), 2)
        self.assertTrue(os.path.exists(f"{self.tmp_dir}/nms.png"))

    def test_suppression_no_suppression_due_to_loose_threshold(self):
        # Test case 3: No suppression when threshold is 0
        contours = [
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[0, 0], [0, 2], [2, 2], [2, 0]],
            [[2, 2], [2, 3], [3, 3], [3, 2]]
        ]
        scores = [0.9, 0.8, 0.7]
        height = 4
        width = 4
        threshold = 1.
        suppressed_contours, suppressed_scores = non_maximum_suppression(
            contours, scores, height, width, threshold)
        self.assertEqual(len(suppressed_contours), 3)
        self.assertEqual(len(suppressed_scores), 3)


if __name__ == '__main__':
    unittest.main()