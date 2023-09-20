import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import filter_size


class TestFilterSize(unittest.TestCase):
    def test_filter_size_0(self):
        # Test case 1: Minimum area is 0, expect no filtering
        pred = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
        min_area = 0
        filtered_pred = filter_size(pred, min_area)
        self.assertEqual(filtered_pred.tolist(), pred.tolist())

    def test_filter_size_5(self):
        # Test case 2: Minimum area is 5, expect filtering of smaller connected components
        pred = np.array([[1, 1, 1, 0, 0],
                         [1, 1, 0, 0, 0],
                         [0, 0, 0, 2, 2],
                         [0, 0, 3, 3, 3]])
        min_area = 5
        expected_pred = np.array([[1, 1, 1, 0, 0],
                                  [1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]])
        filtered_pred = filter_size(pred, min_area)
        self.assertEqual(filtered_pred.tolist(), expected_pred.tolist())

    def test_filter_size_1(self):
        # Test case 3: Minimum area is 1, expect no filtering
        pred = np.array([[1]])
        min_area = 1
        filtered_pred = filter_size(pred, min_area)
        self.assertEqual(filtered_pred.tolist(), pred.tolist())


if __name__ == '__main__':
    unittest.main()