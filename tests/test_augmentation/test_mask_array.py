import unittest
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.augmentation import mask_array

class TestMaskArray(unittest.TestCase):
    
    def test_mask_array_simple(self):
        mask = {
            'images': [{'height': 3, 'width': 3}],
            'annotations': [
                {'segmentation': [[1, 1, 2, 2, 1, 2, 2, 1]]},
                {'segmentation': [[0, 0, 0, 1, 1, 1, 1, 0]]}
            ]
        }
        expected_result = [
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], np.uint8),
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], np.uint8)
        ]
        actual_result = mask_array(mask)
        np.testing.assert_array_equal(actual_result, expected_result)


if __name__ == '__main__':
    unittest.main()