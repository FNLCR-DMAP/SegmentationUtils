import numpy as np
import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import get_intersections

class TestGetIntersections(unittest.TestCase):

    def test_empty_arrays(self):
        gt = np.array([[]])
        pred = np.array([[]])
        with self.assertRaises(IndexError):
            get_intersections(gt, pred)

    def test_custom(self):
        gt = np.array([[0, 1, 1, 3],
                       [0, 1, 1, 0],
                       [0, 0, 0, 2]])
        pred = np.array([[0, 0, 1, 0],
                         [0, 1, 1, 3],
                         [0, 0, 0, 2]])
        expected_result = [
            np.array([[0, 0, 4, 0],
                      [0, 4, 4, 3],
                      [0, 0, 0, 1]]),
            np.array([[0, 2500, 0, 3],
                      [0, 0, 0, 2500],
                      [0, 0, 0, 0]])]
        inference_permuted, gt_xor_inference = get_intersections(gt, pred)
        self.assertTrue(np.array_equal(inference_permuted, expected_result[0]))
        self.assertTrue(np.array_equal(gt_xor_inference, expected_result[1]))

    def test_same_values(self):
        gt = np.array([[0, 1, 1, 3],
                       [0, 1, 1, 0],
                       [0, 0, 0, 2]])
        pred = np.array([[0, 1, 1, 3],
                         [0, 1, 1, 0],
                         [0, 0, 0, 2]])
        expected_result = [
            np.array([[0, 4, 4, 3],
                      [0, 4, 4, 0],
                      [0, 0, 0, 1]]),
            np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])]
        inference_permuted, gt_xor_inference = get_intersections(gt, pred)
        self.assertTrue(np.array_equal(inference_permuted, expected_result[0]))
        self.assertTrue(np.array_equal(gt_xor_inference, expected_result[1]))


if __name__ == '__main__':
    unittest.main()