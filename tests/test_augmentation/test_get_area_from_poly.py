import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.augmentation import get_area_from_poly

class TestGetBboxFromPoly(unittest.TestCase):

    def test_get_area_from_poly1(self):
        # Test case 1: Polygon with 3 points, area should be 0.5
        poly1 = [(0, 0), (1, 0), (0, 1)]
        assert get_area_from_poly(poly1) == 0.5

    def test_get_area_from_poly2(self):
        # Test case 2: Polygon with 4 points, area should be 1
        poly2 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert get_area_from_poly(poly2) == 1.

    def test_get_area_from_poly3(self):
        # Test case 3: Polygon with 5 points, area should be 1
        poly3 = [(0, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)]
        assert get_area_from_poly(poly3) == 1.

    def test_get_area_from_poly4(self):
        # Test case 4: Polygon with 6 points, area should be 1
        poly4 = [(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.5), (0, 1)]
        assert get_area_from_poly(poly4) == 1.

if __name__ == '__main__':
    unittest.main()