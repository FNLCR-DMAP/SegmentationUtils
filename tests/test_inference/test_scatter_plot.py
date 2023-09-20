import unittest
import os

import sys
sys.path.append("../../src")
from pyoseg.inference import scatter_plot


class TestScatterPlot(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_scatter_plot_output(self):
        # Test produced output
        x, y, s = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 1, 2, 3, 4, 5]
        color, xlabel, ylabel = "red", "x", "y"
        name = self.tmp_dir + "/scatter_plot.png"
        scatter_plot(x, y, color, xlabel, ylabel, name, s)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()