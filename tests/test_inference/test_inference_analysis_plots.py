import unittest
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import inference_analysis_plots

class TestInferenceAnalysisPlots(unittest.TestCase):

    def setUp(self):
        self.output_path = ".tmp_inference_plots"
        self.expected_plots = [
            "F1-score_histogram.png",
            "F1-score_vs_gb-ratio.png",
            "F1-score_vs_n_objects.png",
            "F1-score_vs_number-of-cells.png",
            "Precision_histogram.png",
            "Precision_vs_gb-ratio.png",
            "Precision_vs_n_objects.png",
            "Precision_vs_number-of-cells.png",
            "Recall_histogram.png",
            "Recall_vs_gb-ratio.png",
            "Recall_vs_n_objects.png",
            "Recall_vs_number-of-cells.png"
        ]

        # Create output directory
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def tearDown(self):
        # Clean up any created files or directories
        if os.path.exists(self.output_path):
            os.system(f"rm -r {self.output_path}")
            
    def test_inference_analysis_plots_train(self):
        # Test case 2: Non-empty results list
        results = [
            {'image': '000000252219.jpg', 'image_id': 1, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_1-000000252219.png', 'gt_objects': 2, 'detected_objects': 3, 'green_blue_ratio': 0.8648174819979899, 'tp': 1, 'fp': 2, 'fn': 1, 'precision': 0.3333333333333333, 'recall': 0.5, 'f1': 0.4},
            {'image': '000000397133.jpg', 'image_id': 2, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_2-000000397133.png', 'gt_objects': 13, 'detected_objects': 6, 'green_blue_ratio': 1.307068743506462, 'tp': 0, 'fp': 6, 'fn': 13, 'precision': 0, 'recall': 0, 'f1': 0},
            {'image': 'all', 'image_id': '', 'intersection_path': '', 'gt_objects': '', 'detected_objects': '', 'green_blue_ratio': None, 'tp': 1, 'fp': 8, 'fn': 14, 'precision': 0.1111111111111111, 'recall': 0.06666666666666667, 'f1': 0.08333333333333334}
        ]
        inference_analysis_plots(results, self.output_path)
        for e in self.expected_plots:
            self.assertTrue(os.path.exists(f"{self.output_path}/{e}"))
            
    def test_inference_analysis_plots_validation(self):
        # Test case 2: Non-empty results list
        results = [
            {'image': '000000087038.jpg', 'image_id': 1, 'intersection_path': '.tmp_inference_analysis/intersections/intersections_1-000000087038.png', 'gt_objects': 4, 'detected_objects': 29, 'green_blue_ratio': 1.0161126738490682, 'tp': 1, 'fp': 28, 'fn': 3, 'precision': 0.034482758620689655, 'recall': 0.25, 'f1': 0.0606060606060606},
            {'image': 'all', 'image_id': '', 'intersection_path': '', 'gt_objects': '', 'detected_objects': '', 'green_blue_ratio': None, 'tp': 1, 'fp': 28, 'fn': 3, 'precision': 0.034482758620689655, 'recall': 0.25, 'f1': 0.0606060606060606}
        ]
        inference_analysis_plots(results, self.output_path)
        for e in self.expected_plots:
            self.assertTrue(os.path.exists(f"{self.output_path}/{e}"))


if __name__ == "__main__":
    unittest.main()
