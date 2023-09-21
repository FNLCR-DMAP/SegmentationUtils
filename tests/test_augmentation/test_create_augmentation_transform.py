import unittest
import sys
import albumentations as A
import os

sys.path.append("../../src")
from pyoseg.augmentation import create_augmentation_transform, augmentations

class TestCreateAugmentationTransform(unittest.TestCase):
    def test_default_functions(self):
        # Testing with default functions, should return a composed augmentation transform
        expected_transform = str(A.Compose([augmentations[f] for f in augmentations.keys()]))
        transform = create_augmentation_transform()
        self.assertIsNotNone(transform)
        transform = str(transform)
        self.assertEqual(transform, expected_transform)

    def test_selected_functions(self):
        # Testing with selected functions, should return a composed augmentation transform
        selected_functions = ['HorizontalFlip', 'Blur']
        expected_transform = str(A.Compose([augmentations[f] for f in selected_functions]))
        transform = create_augmentation_transform(selected_functions)
        self.assertIsNotNone(transform)
        transform = str(transform)
        self.assertEqual(transform, expected_transform)

    def test_invalid_functions(self):
        # Testing with invalid function names, should raise KeyError
        invalid_functions = ['invalid_function']
        with self.assertRaises(KeyError):
            create_augmentation_transform(invalid_functions)

    def test_invalid_type(self):
        # Testing with invalid function types, should raise KeyError
        invalid_functions = [1]
        with self.assertRaises(KeyError):
            create_augmentation_transform(invalid_functions)

if __name__ == '__main__':
    unittest.main()