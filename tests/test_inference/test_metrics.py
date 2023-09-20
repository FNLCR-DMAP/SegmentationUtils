import unittest

import sys
sys.path.append("../../src")
from pyoseg.inference import metrics


class TestMetrics(unittest.TestCase):
    def test_case1(self):
        tp = 0
        fp = 0
        fn = 0
        expected_output = (0, 0, 0)
        self.assertEqual(metrics(tp, fp, fn), expected_output)

    def test_case2(self):
        tp = 5
        fp = 2
        fn = 3
        expected_output = (0.7142857142857143, 0.625, 0.6666666666666666)
        self.assertAlmostEqual(metrics(tp, fp, fn), expected_output)

    def test_case3(self):
        tp = 10
        fp = 0
        fn = 10
        expected_output = (1.0, 0.5, 0.6666666666666666)
        self.assertAlmostEqual(metrics(tp, fp, fn), expected_output)

    def test_case4(self):
        tp = 0
        fp = 10
        fn = 5
        expected_output = (0.0, 0.0, 0.0)
        self.assertEqual(metrics(tp, fp, fn), expected_output)

    def test_case5(self):
        tp = 10
        fp = 0
        fn = 0
        expected_output = (1.0, 1.0, 1.0)
        self.assertEqual(metrics(tp, fp, fn), expected_output)


if __name__ == '__main__':
    unittest.main()