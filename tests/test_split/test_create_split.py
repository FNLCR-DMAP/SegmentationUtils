import os
import sys
import unittest
from unittest import TestCase

sys.path.append("../../src")
from pyoseg.split import create_split


class CreateSplitTestCase(TestCase):
    
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
        expected_train_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}

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

    def test_augmented_path(self):
        # Test with augmented_path parameter
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        augmented_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/augmentation_folder"
        seed = -1

        expected_split = {
            "train_ids": ["1_aug0", "1_aug1", "1_aug2"],
            "val_ids": ["2_aug2"],
            "test_ids": ["3_aug2"]
        }
        expected_train_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 1}, {"file_name": "test_1.png", "height": 200, "width": 200, "id": 2}, {"file_name": "test_1.png", "height": 200, "width": 200, "id": 3}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}, {"id": 3, "image_id": 2, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 4, "image_id": 2, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}, {"id": 5, "image_id": 3, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 6, "image_id": 3, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}

        split, train_ann, val_ann, test_ann = create_split(annotations_path, output_path=output_path, annotation_suffix=annotation_suffix, train_fraction=train_fraction, validation_fraction=validation_fraction, test_fraction=test_fraction, augmented_path=augmented_path, seed=seed)

        self.assertEqual(set(split['train_ids']), set(expected_split['train_ids']))
        self.assertEqual(set(split['val_ids']), set(expected_split['val_ids']))
        self.assertEqual(set(split['test_ids']), set(expected_split['test_ids']))
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


if __name__ == "__main__":
    unittest.main()
