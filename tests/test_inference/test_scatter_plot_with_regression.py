import unittest
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import scatter_plot_with_regression


class TestScatterPlotWithRegression(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_scatter_plot_with_regression_ooutput(self):
        # Test produced output
        x, y = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]
        color, xlabel, ylabel = "red", "x", "y"
        name = self.tmp_dir + "/scatter_plot_with_regression.png"
        scatter_plot_with_regression(x, y, color, xlabel, ylabel, name)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()
