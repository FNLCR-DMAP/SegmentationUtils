import sys
import unittest
import os

sys.path.append("../../src")
from pyoseg.split import augment_ids
import pyoseg.augmentation as paug
from albumentations import RandomCrop


class TestAugmentIds(unittest.TestCase):

    def setUp(self):
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
        paug.augmentations["RandomCrop"] = self.old_crop
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_augment_ids_all(self):
        train_ids = ['1']
        val_ids = ['2']
        test_ids = ['3']
        split = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + '/test_data/input_folder'
        output_path = self.tmp_dir
        annotation_suffix = '_coco.json'
        
        # Test case 1: ids in all sets
        ids, paths = augment_ids(split, annotations_path, output_path, annotation_suffix, self.augmentation)
        self.assertEqual(set(ids["train_ids"]), set(['1_aug0', '1_aug1']))
        self.assertEqual(set(ids["val_ids"]), set(['2_aug0']))
        self.assertEqual(set(ids["test_ids"]), set(['3_aug0']))
        self.assertTrue(os.path.exists(paths["train_ids"]))
        self.assertTrue(os.path.exists(paths["val_ids"]))
        self.assertTrue(os.path.exists(paths["test_ids"]))

    def test_augment_ids_only_train(self):
        train_ids = ['1', '2']
        val_ids = []
        test_ids = []
        split = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + '/test_data/input_folder'
        output_path = self.tmp_dir
        annotation_suffix = '_coco.json'

        # Test case 2: Only train IDs have augmented files
        with self.assertRaises(FileNotFoundError):
            augment_ids(split, annotations_path, output_path, annotation_suffix, self.augmentation)

    def test_augmentation_only_train(self):
        train_ids = ['1']
        val_ids = ['2']
        test_ids = ['3']
        split = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + '/test_data/input_folder'
        output_path = self.tmp_dir
        annotation_suffix = '_coco.json'

        augmentation = {
            "input_path": os.path.dirname(os.path.realpath(__file__)) + '/test_data/input_images_folder',
            "train": {
                "functions": [
                    "RandomCrop", "HorizontalFlip", "VerticalFlip", "RandomRotate90",
                    "GridDistortion", "Blur", "RandomBrightnessContrast", "RandomGamma"],
                "times": 2},
            "val": {
                "functions": [],
                "times": 0},
            "test": {
                "functions": [],
                "times": 0}
        }
        
        # Test case 3: augmentation only for training set
        ids, paths = augment_ids(split, annotations_path, output_path, annotation_suffix, self.augmentation)
        self.assertEqual(set(ids["train_ids"]), set(['1_aug0', '1_aug1']))
        self.assertEqual(set(ids["val_ids"]), set(['2_aug0']))
        self.assertEqual(set(ids["test_ids"]), set(['3_aug0']))
        self.assertTrue(os.path.exists(paths["train_ids"]))
        self.assertTrue(os.path.exists(paths["val_ids"]))
        self.assertTrue(os.path.exists(paths["test_ids"]))

if __name__ == '__main__':
    unittest.main()