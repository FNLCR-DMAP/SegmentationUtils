import sys
import os
import unittest

sys.path.append("../../src")
import pyoseg.split as ps
from pyoseg.split import create_split_cluster
import pyoseg.augmentation as paug
from albumentations import RandomCrop


class TestCreateSplitCluster(unittest.TestCase):

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

    def test_one_cluster(self):
        # Test with common split
        cluster_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/cluster.csv"
        cluster_column = "Label"
        image_column = "Image"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        seed = 42

        expected_split = {
            "train_ids": ["1"],
            "val_ids": ["2"],
            "test_ids": ["3"]
        }
        expected_train_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}

        split, train_ann, val_ann, test_ann = create_split_cluster(
            cluster_file=cluster_file,
            cluster_column=cluster_column,
            image_column=image_column,
            annotations_path=annotations_path,
            output_path=output_path,
            annotation_suffix=annotation_suffix,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed
        )

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

    def test_more_clusters(self):
        # Test with common split -> names 1,2,3: clusters 1,1,2
        # Test with images on csv file without annotations
        cluster_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/cluster_more.csv"
        cluster_column = "Label"
        image_column = "Image"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        seed = 42

        expected_split = {
            "train_ids": ["2","3"],
            "val_ids": ["1"],
            "test_ids": []
        }
        expected_train_ann = {"images": [{"file_name": "test_2.png", "height": 200, "width": 200, "id": 0}, {"file_name": "test_3.png", "height": 200, "width": 200, "id": 1}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[50, 50]], "area": 1, "bbox": [50, 50, 1, 1], "iscrowd": 0}, {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [], "annotations": [], "categories": []}

        split, train_ann, val_ann, test_ann = create_split_cluster(
            cluster_file=cluster_file,
            cluster_column=cluster_column,
            image_column=image_column,
            annotations_path=annotations_path,
            output_path=output_path,
            annotation_suffix=annotation_suffix,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed
        )

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

    def test_less_images(self):
        # Test with common split -> names 1,3: clusters 1,1
        cluster_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/cluster_less.csv"
        cluster_column = "Label"
        image_column = "Image"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        seed = 42

        expected_split = {
            "train_ids": ["3"],
            "val_ids": ["1"],
            "test_ids": []
        }
        expected_train_ann = {"images": [{"file_name": "test_3.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_val_ann = {"images": [{"file_name": "test_1.png", "height": 200, "width": 200, "id": 0}], "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "segmentation": [[25, 25, 26, 26, 25, 26]], "area": 3, "bbox": [26.0, 26.0, 1.0, 1.0], "iscrowd": 0}, {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[100, 100, 101, 101, 100, 101]], "area": 3, "bbox": [101, 101, 1, 1], "iscrowd": 0}], "categories": [{"id": 1, "name": "Category_1", "supercategory": "Super"}]}
        expected_test_ann = {"images": [], "annotations": [], "categories": []}

        split, train_ann, val_ann, test_ann = create_split_cluster(
            cluster_file=cluster_file,
            cluster_column=cluster_column,
            image_column=image_column,
            annotations_path=annotations_path,
            output_path=output_path,
            annotation_suffix=annotation_suffix,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed
        )

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

    def test_augmentation(self):
        # Test with augmented split
        cluster_file = os.path.dirname(os.path.realpath(__file__)) + "/test_data/cluster.csv"
        cluster_column = "Label"
        image_column = "Image"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/input_folder"
        output_path = os.path.dirname(os.path.realpath(__file__)) + "/test_data/output_folder"
        annotation_suffix = "_coco.json"
        train_fraction = 0.7
        validation_fraction = 0.2
        test_fraction = 0.1
        seed = 42

        expected_split = {
            "train_ids": ["1"],
            "val_ids": ["2"],
            "test_ids": ["3"]
        }
        split, train_ann, val_ann, test_ann = create_split_cluster(
            cluster_file=cluster_file,
            cluster_column=cluster_column,
            image_column=image_column,
            annotations_path=annotations_path,
            output_path=output_path,
            annotation_suffix=annotation_suffix,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            augmentation=self.augmentation
        )

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


if __name__ == '__main__':
    unittest.main()
