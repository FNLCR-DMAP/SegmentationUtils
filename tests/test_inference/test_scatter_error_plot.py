import unittest
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import scatter_error_plot


class TestScatterErrorPlot(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_histogram_plot_output(self):
        # Test produced output
        x, y = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]
        xerr, yerr = [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]
        color, xlabel, ylabel = "red", "x", "y"
        name = self.tmp_dir + "/scatter_error_plot.png"
        scatter_error_plot(x, y, xerr, yerr, color, xlabel, ylabel, name)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()