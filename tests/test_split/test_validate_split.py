import sys
import unittest

sys.path.append("../../src")
from pyoseg.split import validate_split


class TestValidateSplit(unittest.TestCase):
    def test_valid_split(self):
        # Testing valid split fractions
        trainf = 0.6
        valf = 0.2
        testf = 0.2
        self.assertIsNone(validate_split(trainf, valf, testf))

    def test_invalid_split(self):
        # Testing invalid split fractions
        trainf = 0.5
        valf = 0.3
        testf = 0.3
        with self.assertRaises(ValueError):
            validate_split(trainf, valf, testf)


if __name__ == '__main__':
    unittest.main()
