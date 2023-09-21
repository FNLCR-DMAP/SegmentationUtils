import unittest
import json
import os
import unittest
from unittest.mock import patch

import sys
sys.path.append("../../src")
from pyoseg.augmentation import read_annotation_file


class ReadAnnotationFileTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        self.fname = "test.json"
        self.json_dict = {"key": "value"}
        with open(f"{self.tmp_dir}/{self.fname}", "w") as f:
            json.dump(self.json_dict, f)
    
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_read_annotation_file(self):
        # Test case 1: Read a valid annotation file
        result = read_annotation_file(f"{self.tmp_dir}/{self.fname}")
        self.assertEqual(result, self.json_dict)

    def test_read_annotation_file_nonexistent(self):
        # Test case 3: Read a non-existent annotation file
        with self.assertRaises(FileNotFoundError):
            read_annotation_file('nonexistent.json')


if __name__ == '__main__':
    unittest.main()