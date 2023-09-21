import json
import unittest
import numpy as np
from unittest.mock import mock_open, patch
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import get_prediction_annotations

class TestGetPredictionAnnotations(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='[{"image_id": 1, "prediction": "A"}, {"image_id": 2, "prediction": "B"}, {"image_id": 1, "prediction": "C"}]')
    def test_get_prediction_annotations(self, mock_file):
        expected_result = {
            "1": [{"image_id": 1, "prediction": "A"}, {"image_id": 1, "prediction": "C"}],
            "2": [{"image_id": 2, "prediction": "B"}]
        }
        result = get_prediction_annotations("test.json")
        self.assertEqual(result, expected_result)

    @patch('builtins.open', new_callable=mock_open, read_data='[]')
    def test_get_prediction_annotations_empty_file(self, mock_file):
        expected_result = {}
        result = get_prediction_annotations("test.json")
        self.assertEqual(result, expected_result)

    @patch('builtins.open', new_callable=mock_open, read_data='[{"image_id": 1, "prediction": "A"}]')
    def test_get_prediction_annotations_single_prediction(self, mock_file):
        expected_result = {
            "1": [{"image_id": 1, "prediction": "A"}]
        }
        result = get_prediction_annotations("test.json")
        self.assertEqual(result, expected_result)

    def test_get_prediction_annotations_test_file(self):
        predicted_objs = 100
        expected_keys = ['1', '2']
        expected_inner_keys = [
            'bbox', 'segmentation', 'score', 'image_id', 'category_id',
        ]
        result = get_prediction_annotations(os.path.dirname(os.path.realpath(__file__)) + "/../test_data/random_split/coco_instances_train.json")
        self.assertEqual(set(result.keys()), set(expected_keys))
        for e in expected_keys:
            self.assertEqual(len(result[e]), predicted_objs)
            self.assertEqual(set(result[e][0].keys()),set(expected_inner_keys))

    def test_get_prediction_annotations_real_file(self):
        predicted_objs = 2000
        expected_keys = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28',
            '29', '30', '31', '32', '33', '34']
        expected_inner_keys = [
            'bbox', 'segmentation', 'score', 'image_id', 'category_id',
        ]
        result = get_prediction_annotations(os.path.dirname(os.path.realpath(__file__)) + "/../test_data/coco_instances_results.json")
        self.assertEqual(set(result.keys()), set(expected_keys))
        for e in expected_keys:
            self.assertEqual(len(result[e]), predicted_objs)
            self.assertEqual(set(result[e][0].keys()),set(expected_inner_keys))

if __name__ == '__main__':
    unittest.main()