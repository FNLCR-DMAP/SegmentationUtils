import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import get_iou


class TestGetIOU(unittest.TestCase):
    def test_iou_single_object(self):
        inference = np.array([[0, 0, 1, 1],
                              [0, 1, 1, 0],
                              [0, 0, 0, 1]])
        gt = np.array([[0, 0, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 1]])

        iou = get_iou(inference, gt)

        expected_iou = np.array([[0.8]])

        np.testing.assert_array_almost_equal(iou, expected_iou)

    def test_iou_multiple_objects(self):
        inference = np.array([[0, 1, 1, 3],
                              [0, 1, 1, 0],
                              [0, 0, 0, 2]])
        gt = np.array([[0, 0, 1, 0],
                       [0, 1, 1, 3],
                       [0, 0, 0, 2]])

        iou = get_iou(inference, gt, quiet=False)

        expected_iou = np.array([[0.75, 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]])

        np.testing.assert_array_almost_equal(iou, expected_iou)

    def test_iou_missing_objects(self):
        inference = np.array([[0, 1, 1, 0],
                              [0, 1, 1, 0],
                              [0, 0, 0, 2]])
        gt = np.array([[0, 0, 1, 0],
                       [0, 1, 1, 3],
                       [0, 0, 0, 2]])

        iou = get_iou(inference, gt, quiet=False)

        expected_iou = np.array([[0.75, 0.],
                                 [0., 1.],
                                 [0., 0.]])

        np.testing.assert_array_almost_equal(iou, expected_iou)


if __name__ == '__main__':
    unittest.main()
