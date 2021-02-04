#! ../bin/python3
# -*- encoding: utf-8 -*-

import unittest
from images import *


class TestHistogramFunctions(unittest.TestCase):
    def test_get_grayscale_histogram(self):
        testimg = np.array([i for i in range(BITS)], dtype=np.uint8)
        testhist = get_grayscale_histogram(testimg)
        self.assertEqual(testhist, ([1] * BITS))

    def test_get_rgb_histogram(self):
        testrgb = np.zeros((256, 256, 3), dtype=np.uint8)
        c = 0
        for i in range(BITS):
            for j in range(BITS):
                testrgb[i, j] = [c, c, c]
                c += 1
        testhist = get_grayscale_histogram(testimg)
        self.assertEqual(testhist[0], ([256] * BITS))
        self.assertEqual(testhist[1], ([256] * BITS))

    def test_get_hsv_histogram(self):
        pass

