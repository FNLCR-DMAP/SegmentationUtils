import sys
import unittest
import os

sys.path.append("../../src")
from pyoseg.split import create_annotations, TEST_MODE


class TestCreateAnnotations(unittest.TestCase):

    def setUp(self):
        TEST_MODE = True

    def tearDown(self):
        TEST_MODE = False
        
    def test_default_case(self):
        input_folder = os.path.dirname(os.path.realpath(__file__)) + "/test_data/empty_folder"
        output_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder/output.json"
        
        # Call the function
        with self.assertRaises(FileNotFoundError):
            create_annotations(input_folder, output_file)
    
    def test_without_ids(self):
        input_folder = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder/output.json"
        ids = None
        expected_merged_data = {
            "images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}, {"file_name": "test_2.png", "height": 200, "width": 200, "id": 1}, {"file_name": "test_3.png", "height": 200, "width": 200, "id": 2}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}, {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}, {"id": 3, "image_id": 2, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}],
            "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]
        }
        
        # Call the function
        merged_data = create_annotations(input_folder, output_file, ids)
        
        self.assertEqual(merged_data, expected_merged_data)
        self.assertTrue(os.path.exists(output_file))
    
    def test_with_ids(self):
        input_folder = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder/output.json"
        ids = ["1", "2"]
        expected_merged_data = {
            "images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}, {"file_name": "test_2.png", "height": 200, "width": 200, "id": 1}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}, {"id": 2, "image_id": 1, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}],
            "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]
        }
        
        # Call the function
        merged_data = create_annotations(input_folder, output_file, ids)
        
        self.assertEqual(merged_data, expected_merged_data)
        self.assertTrue(os.path.exists(output_file))
    
    def test_with_custom_suffix(self):
        input_folder = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder/output.json"
        annotation_suffix = "_custom.json"
        
        # Call the function
        with self.assertRaises(FileNotFoundError):
            create_annotations(input_folder, output_file, annotation_suffix=annotation_suffix)


if __name__ == '__main__':
    unittest.main()