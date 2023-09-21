import sys
import unittest
import os

sys.path.append("../../src")
from pyoseg.augmentation import get_images_and_masks


class GetImagesAndMasksTestCase(unittest.TestCase):
    def test_get_images_and_masks(self):
        # Check if function returns correct number of images and masks
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/images"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/annotations"
        images, masks, extension = get_images_and_masks(
            data_path, annotations_path)
        
        expected_images = ['000000174482', '000000252219', '000000397133', '000000087038']
        expected_masks = ['000000252219', '000000087038', '000000397133', '000000174482']
        self.assertEqual(len(images), 4)
        self.assertEqual(len(masks), 4)
        self.assertEqual(extension, ".jpg")
        self.assertEqual(set(images), set(expected_images))
        self.assertEqual(set(masks), set(expected_masks))

    def test_get_images_and_missing_masks(self):
        # Check if function returns correct number of images and masks
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/images"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/missing_annotations"
        images, masks, extension = get_images_and_masks(
            data_path, annotations_path)
        
        expected_images = ['000000087038']
        expected_masks = ['000000087038']
        self.assertEqual(len(images), 1)
        self.assertEqual(len(masks), 1)
        self.assertEqual(extension, ".jpg")
        self.assertEqual(set(images), set(expected_images))
        self.assertEqual(set(masks), set(expected_masks))

    def test_get_missing_images_and_masks(self):
        # Check if function returns correct number of images and masks
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/missing_images"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/annotations"
        images, masks, extension = get_images_and_masks(
            data_path, annotations_path)
        
        expected_images = ['000000087038']
        expected_masks = ['000000087038']
        self.assertEqual(len(images), 1)
        self.assertEqual(len(masks), 1)
        self.assertEqual(extension, ".jpg")
        self.assertEqual(set(images), set(expected_images))
        self.assertEqual(set(masks), set(expected_masks))

    def test_get_images_and_masks_no_images(self):
        # Check if function raises FileNotFoundError when no images are found
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/empty_folder"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/annotations"
        with self.assertRaises(FileNotFoundError):
            get_images_and_masks(data_path, annotations_path)

    def test_get_images_and_masks_no_masks(self):
        # Check if function raises FileNotFoundError when no masks are found
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/images"
        annotations_path = os.path.dirname(os.path.realpath(__file__)) + "/../test_data/empty_folder"
        with self.assertRaises(FileNotFoundError):
            get_images_and_masks(data_path, annotations_path)


if __name__ == '__main__':
    unittest.main()