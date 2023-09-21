import numpy as np
import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import sort_objects_by


class TestSortObjectsBy(unittest.TestCase):
    def test_sort_objects_by_ascending(self):
        contours = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        parameter = [10, 20, 30]
        expected_contours = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_parameter = [10, 20, 30]
        
        sorted_contours, sorted_parameter = sort_objects_by(contours, parameter)
        
        self.assertEqual(sorted_contours, expected_contours)
        self.assertEqual(sorted_parameter, expected_parameter)
    
    def test_sort_objects_by_descending(self):
        contours = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
        parameter = [10, 20, 30]
        expected_contours = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_parameter = [30, 20, 10]
        
        sorted_contours, sorted_parameter = sort_objects_by(contours, parameter, ascending=False)
        
        self.assertEqual(sorted_contours, expected_contours)
        self.assertEqual(sorted_parameter, expected_parameter)
    
    def test_sort_objects_by_empty_lists(self):
        contours = []
        parameter = []
        expected_contours = []
        expected_parameter = []
        
        sorted_contours, sorted_parameter = sort_objects_by(contours, parameter)
        
        self.assertEqual(sorted_contours, expected_contours)
        self.assertEqual(sorted_parameter, expected_parameter)


if __name__ == '__main__':
    unittest.main()
