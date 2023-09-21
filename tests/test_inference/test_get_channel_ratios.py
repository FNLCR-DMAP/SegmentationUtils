import numpy as np
import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import get_channel_ratios


class TestChannelRatios(unittest.TestCase):
    def test_ratio_from_values(self):
        channel_1 = [1, 2, 3, 4, 5]
        channel_2 = [6, 7, 8, 9, 10]
        expected_ratio = np.sum(channel_1)/np.sum(channel_2)
        self.assertAlmostEqual(get_channel_ratios(channel_1, channel_2), expected_ratio)

    def test_ratio_from_counts_eq(self):
        channel_1 = [0, 1, 1, 0, 1]
        channel_2 = [1, 1, 0, 1, 0]
        expected_ratio = 1
        self.assertAlmostEqual(get_channel_ratios(channel_1, channel_2, from_counts=True), expected_ratio)

    def test_ratio_from_counts(self):
        channel_1 = [0, 2, 2, 0, 2]
        channel_2 = [1, 1, 0, 1, 0]
        expected_ratio = 1
        self.assertAlmostEqual(get_channel_ratios(channel_1, channel_2, from_counts=True), expected_ratio)

    def test_ratio_from_values_eq(self):
        channel_1 = [0, 2, 1, 0, 2]
        channel_2 = [1, 1, 1, 1, 1]
        expected_ratio = 1
        self.assertAlmostEqual(get_channel_ratios(channel_1, channel_2, from_counts=False), expected_ratio)


if __name__ == '__main__':
    unittest.main()
