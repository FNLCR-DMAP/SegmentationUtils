import numpy as np
import cv2
import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import poly_to_cv


class TestPolyToCV(unittest.TestCase):
    def test_polygon(self):
        poly = [0, 0, 1, 1, 2, 2]
        height = 3
        width = 3
        expected_output = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(poly_to_cv(poly, height, width), expected_output))

    def test_empty_polygon(self):
        poly = []
        height = 3
        width = 3
        expected_output = np.zeros((height, width), dtype=np.uint8)
        self.assertTrue(np.array_equal(poly_to_cv(poly, height, width), expected_output))

    def test_large_polygon(self):
        poly = [0, 0, 0, 1, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1, 2, 2]
        height = 5
        width = 5
        expected_output = np.array([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 0, 0],
                                   [1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(poly_to_cv(poly, height, width), expected_output))


if __name__ == "__main__":
    unittest.main()
