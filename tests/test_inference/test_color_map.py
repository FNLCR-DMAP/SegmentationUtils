import unittest
import numpy as np
from matplotlib.colors import ListedColormap

import sys
sys.path.append("../../src")
from pyoseg.inference import color_map


class ColorMapTestCase(unittest.TestCase):
    def test_color_map(self):
        # Test if the returned colormap is an instance of ListedColormap
        cmap = color_map()
        self.assertTrue(isinstance(cmap, ListedColormap))

        # Test if the colormap contains 256 colors
        self.assertTrue(cmap.N == 256)

        # Test if the first color in the colormap is black
        self.assertTrue(np.array_equal(cmap.colors[0], [0, 0, 0, 1]))

        # Test if the last color in the colormap is magneta
        self.assertTrue(np.array_equal(cmap.colors[-1], [1, 0, 1, 1]))


if __name__ == '__main__':
    unittest.main()