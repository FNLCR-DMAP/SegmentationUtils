import os
import sys
import unittest
from unittest import TestCase

sys.path.append("../../src")
import pyoseg.split as ps
from pyoseg.split import create_split
import pyoseg.augmentation as paug
from albumentations import RandomCrop


class CreateSplitTestCase(TestCase):

    def setUp(self):
        ps.TEST_MODE = True
        self.old_crop = paug.augmentations["RandomCrop"]
        paug.augmentations["RandomCrop"] =  RandomCrop(p=1., height=100, width=100)
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        self.augmentation = {
            "input_path": os.path.dirname(os.path.realpath(__file__)) + '/test_data/input_images_folder',
            "train": {
                "functions": [
                    "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90",
                    "GridDistortion", "Blur", "RandomBrightnessContrast", "RandomGamma"],
                "times": 2},
            "val": {
                "functions": ["RandomCrop"],
                "times": 1},
            "test": {
                "functions": ["RandomCrop"],
                "times": 1}
        }

    def tearDown(self):
        ps.TEST_MODE = False
        paug.augmentations["RandomCrop"] = self.old_crop
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")
    
    def test_default_parameters(self):
        # Test with default parameters
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        seed = -1

        expected_split = {
            "train_ids": ["1"],
            "val_ids": ["2"],
            "test_ids": ["3"]
        }
        expected_train_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}

        split, train_ann, val_ann, test_ann = create_split(annotations_path, output_path, seed=seed)

        self.assertEqual(len(split['train_ids']), len(expected_split['train_ids']))
        self.assertEqual(len(split['val_ids']), len(expected_split['val_ids']))
        self.assertEqual(len(split['test_ids']), len(expected_split['test_ids']))
        self.assertEqual(train_ann, expected_train_ann)
        self.assertEqual(val_ann, expected_val_ann)
        self.assertEqual(test_ann, expected_test_ann)
        self.assertTrue(os.path.exists(f"{output_path}/split.json"))
        self.assertTrue(os.path.exists(f"{output_path}/train_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/validation_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/test_annotations.json"))

    def test_invalid_fractions(self):
        # Test with invalid fractions
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.5
        validation_fraction = 0.3
        test_fraction = 0.3

        with self.assertRaises(ValueError):
            create_split(annotations_path, output_path=output_path, annotation_suffix=annotation_suffix, train_fraction=train_fraction, 
                         validation_fraction=validation_fraction, test_fraction=test_fraction)
            
    def test_augmentation(self):
        # Test with augmented_path parameter
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = self.tmp_dir
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        seed = -1

        expected_split = {
            "train_ids": ["1"],
            "val_ids": ["2"],
            "test_ids": ["3"]
        }
        split, train_ann, val_ann, test_ann = create_split(
            annotations_path, output_path=output_path,
            annotation_suffix=annotation_suffix,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            augmentation=self.augmentation,
            seed=seed)

        self.assertEqual(set(split['train_ids']), set(expected_split['train_ids']))
        self.assertEqual(set(split['val_ids']), set(expected_split['val_ids']))
        self.assertEqual(set(split['test_ids']), set(expected_split['test_ids']))
        self.assertEqual(len(train_ann["images"]), self.augmentation["train"]["times"])
        self.assertEqual(len(val_ann["images"]), self.augmentation["val"]["times"])
        self.assertEqual(len(test_ann["images"]), self.augmentation["test"]["times"])
        self.assertTrue(os.path.exists(f"{output_path}/split.json"))
        self.assertTrue(os.path.exists(f"{output_path}/train_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/validation_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/test_annotations.json"))


if __name__ == "__main__":
    unittest.main()
