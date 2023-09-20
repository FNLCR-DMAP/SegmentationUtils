import unittest
import cv2
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import polygon_from_mask

class TestPolygonFromMask(unittest.TestCase):

    def test_no_valid_polygons(self):
        maskedArr = np.zeros((100, 100), dtype=np.uint8)
        result = polygon_from_mask(maskedArr)
        self.assertEqual(result, [None])

    def test_single_valid_polygon(self):
        maskedArr = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(maskedArr, [np.array([[10, 10], [20, 10], [15, 20]])], -1, (255), 1)
        result = polygon_from_mask(maskedArr)
        expected_polygon = [
            10, 10, 10, 11, 11, 12, 11, 13, 12, 14, 12, 15, 13, 16, 13, 17, 14,
            18, 14, 19, 15, 20, 15, 19, 16, 18, 16, 17, 17, 16, 17, 15, 18, 14,
            18, 13, 19, 12, 19, 11, 20, 10]
        expected_bounding_rect = [10, 10, 11, 11]
        expected_area = 45
        np.testing.assert_array_equal(result[0], expected_polygon)
        np.testing.assert_array_equal(result[1], expected_bounding_rect)
        self.assertEqual(result[2], expected_area)

    def test_multiple_valid_polygons(self):
        maskedArr = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(maskedArr, [np.array([[10, 10], [20, 10], [15, 20]]),
                                       np.array([[30, 30], [40, 30], [35, 40]])], -1, (255), 1)
        result = polygon_from_mask(maskedArr)
        expected_polygon = [
            30, 30, 30, 31, 31, 32, 31, 33, 32, 34, 32, 35, 33, 36, 33, 37, 34,
            38, 34, 39, 35, 40, 35, 39, 36, 38, 36, 37, 37, 36, 37, 35, 38, 34,
            38, 33, 39, 32, 39, 31, 40, 30]
        expected_bounding_rect = [10, 10, 31, 31]
        expected_area = 90
        np.testing.assert_array_equal(result[0], expected_polygon)
        np.testing.assert_array_equal(result[1], expected_bounding_rect)
        self.assertEqual(result[2], expected_area)

    def test_simple_polygon(self):
        maskedArr = np.zeros((4, 4), dtype=np.uint8)
        cv2.drawContours(
            maskedArr, [np.array([[0, 0], [0, 1], [1, 0], [1, 1]])], -1, (255), 1)
        result = polygon_from_mask(maskedArr)
        expected_polygon = [0, 0, 0, 1, 1, 1, 1, 0]
        expected_bounding_rect = [0, 0, 2, 2]
        expected_area = 1
        np.testing.assert_array_equal(result[0], expected_polygon)
        np.testing.assert_array_equal(result[1], expected_bounding_rect)
        self.assertEqual(result[2], expected_area)

if __name__ == '__main__':
    unittest.main()