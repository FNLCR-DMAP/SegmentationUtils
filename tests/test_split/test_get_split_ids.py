import sys
import unittest
from unittest.mock import mock_open, patch
import os

sys.path.append("../../src")
from pyoseg.split import get_split_ids


class TestGetSplitIds(unittest.TestCase):
    def test_get_split_ids(self):
        expected_result = [(1, "test_1.png"), (2, "test_2.png")]
        result = get_split_ids(os.path.dirname(os.path.realpath(__file__)) + "/test_data/train_annotations.json")
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()