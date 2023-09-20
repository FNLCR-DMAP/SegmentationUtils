import unittest
import os
import cv2
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.augmentation import augment_data

class TestAugmentData(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_augment_data_default(self):
        # Test case 1: Test augmentation with valid parameters
        data_path = "../test_data/images"
        annotations_path = "../test_data/annotations"
        output_path = self.tmp_dir
        annotation_suffix = "_coco.json"
        times = 2
        
        augment_data(
            data_path, annotations_path, output_path,
            annotation_suffix=annotation_suffix, times=times)
        
        # Check if the augmented images and masks are saved in the output path
        self.assertTrue(os.path.exists(output_path))
        for i in range(times+1):
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000087038_aug{i}.jpg")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000087038_aug{i}{annotation_suffix}")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000174482_aug{i}.jpg")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000174482_aug{i}{annotation_suffix}")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000252219_aug{i}.jpg")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000252219_aug{i}{annotation_suffix}")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000397133_aug{i}.jpg")))
            self.assertTrue(os.path.exists(os.path.join(output_path, f"000000397133_aug{i}{annotation_suffix}")))

def idk():
    def test_augment_data(self):
        # Test case 2: Test augmentation with no masks
        data_path = "data/images"
        annotations_path = "data/masks"
        output_path = "output"
        annotation_suffix = "_coco.json"
        functions = ["RandomCrop"]
        times = 10
        
        # Remove the masks from the annotations path
        
        augment_data(
            data_path, annotations_path, output_path,
            annotation_suffix=annotation_suffix, functions=functions, times=times)
        
        # Check if the augmented images are saved in the output path
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.exists(os.path.join(output_path, "image1_aug0.png")))
        self.assertTrue(os.path.exists(os.path.join(output_path, "image1_aug1.png")))
        # Add more assertions for other augmented images
    
    def test_augment_data(self):
        # Test case 3: Test augmentation with invalid augmentation function
        data_path = "data/images"
        annotations_path = "data/masks"
        output_path = "output"
        annotation_suffix = "_coco.json"
        functions = ["InvalidFunction"]
        times = 10
        
        # Remove the masks from the annotations path
        
        # Check if the augmentation function is invalid
        with self.assertRaises(AssertionError):
            augment_data(
                data_path, annotations_path, output_path,
                annotation_suffix=annotation_suffix, functions=functions, times=times)
        
        # Check if no augmented images are saved in the output path
        self.assertTrue(os.path.exists(output_path))
        self.assertFalse(os.path.exists(os.path.join(output_path, "image1_aug0.png")))
        self.assertFalse(os.path.exists(os.path.join(output_path, "image1_aug1.png")))
        # Add more assertions for other augmented images
    
    def test_augment_data(self):
        # Test case 4: Test augmentation with invalid data and annotations paths
        data_path = "invalid_path"
        annotations_path = "invalid_path"
        output_path = "output"
        annotation_suffix = "_coco.json"
        functions = ["RandomCrop"]
        times = 10
        
        # Check if the data and annotations paths are invalid
        with self.assertRaises(FileNotFoundError):
            augment_data(
                data_path, annotations_path, output_path,
                annotation_suffix=annotation_suffix, functions=functions, times=times)
        
        # Check if no augmented images are saved in the output path
        self.assertTrue(os.path.exists(output_path))
        self.assertFalse(os.path.exists(os.path.join(output_path, "image1_aug0.png")))
        self.assertFalse(os.path.exists(os.path.join(output_path, "image1_aug1.png")))
        # Add more assertions for other augmented images
        
if __name__ == '__main__':
    unittest.main()