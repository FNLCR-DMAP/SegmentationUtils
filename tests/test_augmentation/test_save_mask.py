import unittest
import json
import os

import sys
sys.path.append("../../src")
from pyoseg.augmentation import save_mask


class SaveMaskTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
    
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_save_mask(self):
        mask_path = f"{self.tmp_dir}/mask.json"
        mask = {'data': [1, 2, 3]}
        save_mask(mask_path, mask)
        self.assertTrue(os.path.exists(mask_path))
        read_mask = json.load(open(mask_path, 'r'))
        self.assertEqual(mask, read_mask)


if __name__ == '__main__':
    unittest.main()