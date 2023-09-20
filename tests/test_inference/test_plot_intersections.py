import unittest
import os
import numpy as np

import sys
sys.path.append("../../src")
from pyoseg.inference import plot_intersections


class TestPlotIntersections(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = ".tmp_test"
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            os.system(f"rm -r {self.tmp_dir}")

    def test_plot_intersections_output(self):
        # Test produced output
        image = np.array([[(0,0,0), (2,3,2), (3,2,3), (15,14,15)],
                          [(0,0,0), (3,2,3), (2,3,2), (0,0,0)],
                          [(0,0,0), (0,0,0), (0,0,0), (25,24,25)]])
        gt = np.array([[0, 1, 1, 3],
                       [0, 1, 1, 0],
                       [0, 0, 0, 2]])
        pred = np.array([[0, 0, 1, 0],
                         [0, 1, 1, 3],
                         [0, 0, 0, 2]])
        gt_per = np.array([[0, 0, 4, 0],
                           [0, 4, 4, 3],
                           [0, 0, 0, 1]]),
        gt_xor = np.array([[0, 2500, 0, 3],
                           [0, 0, 0, 2500],
                           [0, 0, 0, 0]])
        name = self.tmp_dir + "/intersections.png"
        plot_intersections(gt, gt_per, gt_xor, image, name)
        self.assertTrue(os.path.exists(name))


if __name__ == '__main__':
    unittest.main()