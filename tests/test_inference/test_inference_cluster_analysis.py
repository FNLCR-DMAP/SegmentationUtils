import unittest
import pandas as pd
import numpy as np
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import inference_cluster_analysis


class TestInferenceClusterAnalysis(unittest.TestCase):

    def setUp(self):
        self.output_path = ".tmp_cluster_analysis"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.system(f"rm -r {self.output_path}")

    def test_inference_cluster_analysis(self):
        # Create dummy data for testing
        results = [
            {'image': '000000252219.jpg', 'image_id': 1, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_1-000000252219.png', 'gt_objects': 2, 'detected_objects': 3, 'green_blue_ratio': 0.8648174819979899, 'tp': 1, 'fp': 2, 'fn': 1, 'precision': 0.3333333333333333, 'recall': 0.5, 'f1': 0.4},
            {'image': '000000397133.jpg', 'image_id': 2, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_2-000000397133.png', 'gt_objects': 13, 'detected_objects': 6, 'green_blue_ratio': 1.307068743506462, 'tp': 0, 'fp': 6, 'fn': 13, 'precision': 0, 'recall': 0, 'f1': 0},
            {'image': '000000397134.jpg', 'image_id': 3, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_2-000000397134.png', 'gt_objects': 2, 'detected_objects': 3, 'green_blue_ratio': 0.8648174819979899, 'tp': 1, 'fp': 2, 'fn': 1, 'precision': 0.3333333333333333, 'recall': 0.5, 'f1': 0.4},
            {'image': 'all', 'image_id': '', 'intersection_path': '', 'gt_objects': '', 'detected_objects': '', 'green_blue_ratio': None, 'tp': 1, 'fp': 8, 'fn': 14, 'precision': 0.1111111111111111, 'recall': 0.06666666666666667, 'f1': 0.08333333333333334}
        ]
        cluster_file = '../test_data/cluster.csv'
        cluster_column = 'cluster'
        image_column = 'image'

        # Call the function
        df = inference_cluster_analysis(results, cluster_file, cluster_column, image_column, self.output_path)

        # Check the output
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(df.columns.tolist(), ['cluster', 'precision', 'recall', 'f1'])
        self.assertEqual(df['cluster'].tolist(), [0, 1])
        np.testing.assert_array_almost_equal(df['precision'].tolist(), [0.111111, results[2]['precision']])
        np.testing.assert_array_almost_equal(df['recall'].tolist(), [0.066667, results[2]['recall']])
        np.testing.assert_array_almost_equal(df['f1'].tolist(), [0.083333, results[2]['f1']])


if __name__ == '__main__':
    unittest.main()