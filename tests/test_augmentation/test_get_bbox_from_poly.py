import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.augmentation import get_bbox_from_poly

class TestGetBboxFromPoly(unittest.TestCase):

    def test_get_bbox_from_square_poly(self):
        # Test case 1: Square polygon
        poly = [0, 0, 1, 0, 1, 1, 0, 1]
        expected_bbox = [0, 0, 1, 1]
        self.assertEqual(get_bbox_from_poly(poly), expected_bbox)

    def test_get_bbox_from_rect_poly(self):
        # Test case 2: Rectangle polygon
        poly = [0, 0, 2, 0, 2, 1, 0, 1]
        expected_bbox = [0, 0, 2, 1]
        self.assertEqual(get_bbox_from_poly(poly), expected_bbox)

    def test_get_bbox_from_triang_poly(self):
        # Test case 3: Triangle polygon
        poly = [0, 0, 1, 0, 0.5, 1]
        expected_bbox = [0, 0, 1, 1]
        self.assertEqual(get_bbox_from_poly(poly), expected_bbox)

    def test_get_bbox_from_single_poly(self):
        # Test case 4: Single point polygon
        poly = [0, 0]
        expected_bbox = [0, 0, 0, 0]
        self.assertEqual(get_bbox_from_poly(poly), expected_bbox)

    def test_get_bbox_from_empty_poly(self):
        # Test case 5: Empty polygon
        poly = []
        expected_bbox = [0, 0, 0, 0]
        self.assertEqual(get_bbox_from_poly(poly), expected_bbox)

if __name__ == '__main__':
    unittest.main()