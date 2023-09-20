import numpy as np
import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import get_permuted_lookup


class TestGetPermutedLookup(unittest.TestCase):

    def test_get_permuted_lookup1(self):
        # Test case 1: Empty list of instance ids
        ids = []
        with self.assertRaises(ValueError):
            get_permuted_lookup(ids)

    def test_get_permuted_lookup2(self):
        # Test case 2: List of instance ids with only one element
        ids = [1]
        result = get_permuted_lookup(ids)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(len(result) == 2)
        self.assertTrue(result[0] == 0)

    def test_get_permuted_lookup3(self):
        # Test case 3: List of instance ids with multiple elements
        ids = [1, 2, 3, 4, 5]
        result = get_permuted_lookup(ids)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(len(result) == np.max(ids) + 1)
        self.assertTrue(result[0] == 0)

    def test_get_permuted_lookup4(self):
        # Test case 4: List of instance ids with large values
        ids = [1000, 2000, 3000]
        result = get_permuted_lookup(ids)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(len(result) == np.max(ids) + 1)
        self.assertTrue(result[0] == 0)

    def test_get_permuted_lookup5(self):
        # Test case 5: List of instance ids with duplicate values
        ids = [1, 2, 3, 4, 5, 1, 2, 3]
        result = get_permuted_lookup(ids)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(len(result) == np.max(ids) + 1)
        self.assertTrue(result[0] == 0)


if __name__ == '__main__':
    unittest.main()