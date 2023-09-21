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
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/images"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/annotations"
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

        
if __name__ == '__main__':
    unittest.main()