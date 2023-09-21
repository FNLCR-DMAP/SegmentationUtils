import numpy as np
import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import precision_at


class TestPrecisionAt(unittest.TestCase):

    def setUp(self):
        self.iou = np.array([[0.75, 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 0.]])

    def test_precision_at_threshold_0_7(self):
        # Testing precision at threshold 0.5
        true_positives, false_positives, false_negatives = \
            precision_at(0.7, self.iou, quiet=True)
        np.testing.assert_array_equal(true_positives, [True, True, False])
        np.testing.assert_array_equal(false_positives, [False, False, True])
        np.testing.assert_array_equal(false_negatives, [False, False, True])

    def test_precision_at_threshold_0_8(self):
        # Testing precision at threshold 0.8
        true_positives, false_positives, false_negatives = \
            precision_at(0.8, self.iou, quiet=True)
        np.testing.assert_array_equal(true_positives, [False, True, False])
        np.testing.assert_array_equal(false_positives, [True, False, True])
        np.testing.assert_array_equal(false_negatives, [True, False, True])


if __name__ == '__main__':
    unittest.main()
