import json
import unittest
import numpy as np
from unittest.mock import mock_open, patch

import sys
sys.path.append("../../src")
from pyoseg.inference import get_gt_annotations

class TestGetGtAnnotations(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='{"images": [{"id": 1}, {"id": 2}], "annotations": [{"image_id": 1, "segmentation": "A"}, {"image_id": 2, "segmentation": "B"}]}')
    def test_get_gt_annotations(self, mock_file):
        expected_result = {
            '1': {
                'images': [{'id': 1}],
                'annotations': [{'image_id': 1, 'segmentation': 'A'}]},
            '2': {
                'images': [{'id': 2}],
                'annotations': [{'image_id': 2, 'segmentation': 'B'}]}}
        result = get_gt_annotations("test.json", [1, 2])
        self.assertEqual(result, expected_result)

    @patch('builtins.open', new_callable=mock_open, read_data='{"images": [{"id": 1}, {"id": 2}], "annotations": [{"image_id": 1, "segmentation": "A"}, {"image_id": 2, "segmentation": "B"}]}')
    def test_get_gt_annotations_no_ids(self, mock_file):
        expected_result = {}
        result = get_gt_annotations("test.json", ["1", 3])
        self.assertEqual(result, expected_result)

    @patch('builtins.open', new_callable=mock_open, read_data='{"images": [{"id": 1}, {"id": 2}], "annotations": [{"image_id": 1, "segmentation": "A"}, {"image_id": 2, "segmentation": "B"}]}')
    def test_get_gt_annotations_ids(self, mock_file):
        expected_result = {
            '2': {
                'images': [{'id': 2}],
                'annotations': [{'image_id': 2, 'segmentation': 'B'}]}}
        result = get_gt_annotations("test.json", [2])
        self.assertEqual(result, expected_result)

    @patch('builtins.open', new_callable=mock_open, read_data='[]')
    def test_get_gt_annotations_empty_file(self, mock_file):
        expected_result = {}
        result = get_gt_annotations("test.json", [])
        self.assertEqual(result, expected_result)

    def test_get_gt_annotations_test_file(self):
        expected_keys = ["1", "2"]
        expected_inner_keys = ["images", "annotations"]
        result = get_gt_annotations("../test_data/random_split/train_annotations.json", [1, 2, 3, 4])
        
        self.assertEqual(set(result.keys()), set(expected_keys))
        for e in expected_keys:
            self.assertEqual(set(result[e].keys()), set(expected_inner_keys))
        self.assertEqual(len(result['1']['annotations']), 7)
        self.assertEqual(len(result['2']['annotations']), 14)

if __name__ == '__main__':
    unittest.main()