import sys
import unittest

sys.path.append("../../src")
from pyoseg.split import augment_ids

class TestAugmentIds(unittest.TestCase):
    def test_augment_ids_all(self):
        train_ids = ['1']
        val_ids = ['2']
        test_ids = ['3']
        augmented_path = 'test_data/augmentation_folder'
        annotation_suffix = '_coco.json'
        
        # Test case 1: ids in all sets
        aug_train_ids, aug_val_ids, aug_test_ids = augment_ids(train_ids, val_ids, test_ids, augmented_path, annotation_suffix)
        self.assertEqual(aug_train_ids, ['1_aug0', '1_aug1', '1_aug2'])
        self.assertEqual(aug_val_ids, ['2_aug2'])
        self.assertEqual(aug_test_ids, ['3_aug2'])
    
    def test_augment_ids_only_train(self):
        train_ids = ['1', '2']
        val_ids = []
        test_ids = []
        augmented_path = 'test_data/augmentation_folder'
        annotation_suffix = '_coco.json'

        # Test case 2: Only train IDs have augmented files
        aug_train_ids, aug_val_ids, aug_test_ids = augment_ids(train_ids, val_ids, test_ids, augmented_path, annotation_suffix)
        self.assertEqual(aug_train_ids, ['1_aug0', '1_aug1', '1_aug2', '2_aug0','2_aug1', '2_aug2'])
        self.assertEqual(aug_val_ids, [])
        self.assertEqual(aug_test_ids, [])
    
    def test_augment_ids_not_only_train(self):
        train_ids = ['1', '2', '3']
        val_ids = ['1', '2']
        test_ids = ['3']
        augmented_path = 'test_data/augmentation_folder'
        annotation_suffix = '_coco.json'

        # Test case 3: Only train and val IDs have augmented files, with aug_train_only=False
        aug_train_ids, aug_val_ids, aug_test_ids = augment_ids(train_ids, val_ids, test_ids, augmented_path, annotation_suffix, aug_train_only=False)
        self.assertEqual(aug_train_ids, ['1_aug0', '1_aug1', '1_aug2', '2_aug0', '2_aug1', '2_aug2', '3_aug0', '3_aug1', '3_aug2'])
        self.assertEqual(aug_val_ids, ['1_aug0', '1_aug1', '1_aug2', '2_aug0','2_aug1', '2_aug2'])
        self.assertEqual(aug_test_ids, ['3_aug0', '3_aug1', '3_aug2'])

if __name__ == '__main__':
    unittest.main()