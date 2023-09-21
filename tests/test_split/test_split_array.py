import sys
import unittest

sys.path.append("../../src")
from pyoseg.split import split_array


class TestSplitArray(unittest.TestCase):
    def test_empty_array(self):
        train, val, test = split_array([], 0, 0.2, 0.1)
        self.assertEqual(train, [])
        self.assertEqual(val, [])
        self.assertEqual(test, [])

    def test_single_element_array(self):
        train, val, test = split_array([1], 1, 0.2, 0.1)
        self.assertEqual(train, [1])
        self.assertEqual(val, [])
        self.assertEqual(test, [])

    def test_two_element_array(self):
        train, val, test = split_array([1, 2], 2, 0.2, 0.1)
        self.assertEqual(train, [1])
        self.assertEqual(val, [2])
        self.assertEqual(test, [])

    def test_three_element_array(self):
        train, val, test = split_array([1, 2, 3], 3, 0.2, 0.1)
        self.assertEqual(train, [1])
        self.assertEqual(val, [2])
        self.assertEqual(test, [3])

    def test_multiple_element_array(self):
        train, val, test = split_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, 0.2, 0.1)
        self.assertEqual(train, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(val, [8, 9])
        self.assertEqual(test, [10])


if __name__ == '__main__':
    unittest.main()