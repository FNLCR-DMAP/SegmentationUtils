import unittest
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import histogram_plot


class TestHistogramPlot(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_histogram_plot_output(self):
        # Test produced output
        data = [1, 2, 3, 4, 5, 6]
        color, xlabel, ylabel = "red", "x", "y"
        name = self.tmp_dir + "/histogram_plot.png"
        histogram_plot(data, color, xlabel, ylabel, name)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()
