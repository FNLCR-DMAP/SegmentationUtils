import unittest
from PIL import Image
import numpy as np
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import get_image_channels


class TestGetImageChannels(unittest.TestCase):
    def test_get_image_channels_img1(self):
        # Test case 1: Test with a valid image file path
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../test_data/images/000000397133.jpg'
        r, g, b = get_image_channels(file_path)
        
        # Assert that the returned values are instances of PIL.Image.Image
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(g, Image.Image)
        self.assertIsInstance(b, Image.Image)
        self.assertEqual(np.sum(np.array(r).astype(bool)), 272252)
        self.assertEqual(np.sum(np.array(g).astype(bool)), 271891)
        self.assertEqual(np.sum(np.array(b).astype(bool)), 269118)
        
    def test_get_image_channels_img2(self):
        # Test case 2: Test with a different valid image file path
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../test_data/images//000000252219.jpg'
        r, g, b = get_image_channels(file_path)
        
        # Assert that the returned values are instances of PIL.Image.Image
        self.assertIsInstance(r, Image.Image)
        self.assertIsInstance(g, Image.Image)
        self.assertIsInstance(b, Image.Image)
        self.assertEqual(np.sum(np.array(r).astype(bool)), 220813)
        self.assertEqual(np.sum(np.array(g).astype(bool)), 268114)
        self.assertEqual(np.sum(np.array(b).astype(bool)), 271029)
        
    def test_get_image_channels_invalid(self):
        # Test case 3: Test with an invalid image file path
        file_path = 'path/to/non_existent_image.jpg'
        # Assert that a FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            get_image_channels(file_path)


if __name__ == '__main__':
    unittest.main()
