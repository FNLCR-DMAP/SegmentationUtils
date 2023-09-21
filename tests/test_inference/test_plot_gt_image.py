import unittest
import os
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import plot_gt_image


class TestPlotGtImage(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_plot_gt_image_output(self):
        # Test produced output
        gt = np.array([[0, 1, 1, 3],
                       [0, 1, 1, 0],
                       [0, 0, 0, 2]])
        image = np.array([[(0,0,0), (2,3,2), (3,2,3), (15,14,15)],
                          [(0,0,0), (3,2,3), (2,3,2), (0,0,0)],
                          [(0,0,0), (0,0,0), (0,0,0), (25,24,25)]])
        name = self.tmp_dir + "/gt_image.png"
        plot_gt_image(image, gt, name)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()
