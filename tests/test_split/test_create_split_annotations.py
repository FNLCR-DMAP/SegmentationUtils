import os
import sys
import unittest
from unittest.mock import patch

sys.path.append("../../src")
from pyoseg.split import create_split_annotations


class TestCreateSplitAnnotations(unittest.TestCase):
    def test_split_sizes(self):
        train_ids = ["1", "2"]
        val_ids = ["2"]
        test_ids = ["3"]
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder/"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder/"
        annotation_suffix = "_coco.json"

        expected_split = {
            "train_ids": ["1","2"],
            "val_ids": ["2"],
            "test_ids": ["3"]
        }
        expected_train_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 1}, {"file_name": "test_2.png", "height": 200, "width": 200, "id": 2}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}, {"id": 3, "image_id": 2, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        
        # Call the function
        split, train_ann, val_ann, test_ann = create_split_annotations(train_ids, val_ids, test_ids, annotations_path, output_path, annotation_suffix)
        
        # Assertions
        self.assertEqual(split, expected_split)
        self.assertEqual(train_ann, expected_train_ann)
        self.assertEqual(val_ann, expected_val_ann)
        self.assertEqual(test_ann, expected_test_ann)
        self.assertTrue(os.path.exists(f"{output_path}/split.json"))
        self.assertTrue(os.path.exists(f"{output_path}/train_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/validation_annotations.json"))
        self.assertTrue(os.path.exists(f"{output_path}/test_annotations.json"))


if __name__ == '__main__':
    unittest.main()