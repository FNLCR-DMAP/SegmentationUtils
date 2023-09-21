import numpy as np
import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.augmentation import poly_to_cv, cv_to_poly


class TestCVToPoly(unittest.TestCase):
    def test_cv_to_poly_empty(self):
        # Case when cv_mask is all zeros
        cv_mask = np.zeros((10, 10))
        self.assertTrue(cv_to_poly(cv_mask) is None)

    def test_cv_to_poly_less_6_points(self):
        # Case when cv_mask has a contour with less than 6 points
        cv_mask = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
        expected_result = [0, 0, 0, 1, 1, 1, 1, 0]
        self.assertTrue(np.array_equal(cv_to_poly(cv_mask), expected_result))

    def test_cv_to_poly_custom(self):
        # Case when cv_mask has a contour with more than 6 points
        cv_mask = np.array([[1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
        expected_result = [0, 0, 0, 2, 2, 2, 2, 0]
        self.assertTrue(np.array_equal(cv_to_poly(cv_mask), expected_result))

    def test_cv_to_poly_complete(self):
        poly = [0, 0, 0, 2, 2, 2, 2, 0]
        height = 5
        width = 5
        cv = poly_to_cv(poly, height, width)
        result = cv_to_poly(cv)
        self.assertTrue(np.array_equal(result, poly))


if __name__ == "__main__":
    unittest.main()