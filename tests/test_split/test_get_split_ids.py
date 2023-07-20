import sys
import json
import unittest
from unittest.mock import mock_open, patch

sys.path.append("../../src")
from pyoseg.split import get_split_ids

class TestGetSplitIds(unittest.TestCase):
    def test_get_split_ids(self):
        expected_result = [(1, "test_1.png"), (2, "test_2.png")]
        result = get_split_ids("test_data/train_annotations.json")
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()