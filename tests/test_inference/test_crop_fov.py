import numpy as np
import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import crop_fov


class CropFovTestCase(unittest.TestCase):
    def test_crop_fov_top_left(self):
        # Test case 1: Crop a region from the top-left corner
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cropped_im = crop_fov(im, 0, 1, 0, 1)
        expected_result = np.array([[1]])
        np.testing.assert_array_equal(cropped_im, expected_result)

    def test_crop_fov_center(self):
        # Test case 2: Crop a region from the bottom-right corner
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cropped_im = crop_fov(im, 1, 2, 1, 2)
        expected_result = np.array([[5]])
        np.testing.assert_array_equal(cropped_im, expected_result)

    def test_crop_fov_bottom_right(self):
        # Test case 2: Crop a region from the bottom-right corner
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cropped_im = crop_fov(im, 1, 3, 1, 3)
        expected_result = np.array([[5, 6], [8, 9]])
        np.testing.assert_array_equal(cropped_im, expected_result)

    def test_crop_fov_entire(self):
        # Test case 3: Crop a region covering the entire image
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cropped_im = crop_fov(im, 0, 3, 0, 3)
        expected_result = im
        np.testing.assert_array_equal(cropped_im, expected_result)


if __name__ == '__main__':
    unittest.main()
