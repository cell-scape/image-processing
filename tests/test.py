#! ../bin/python3
# -*- encoding: utf-8 -*-

import unittest
from images import *


class ImagesTestSuite(unittest.TestSuite):
    pass


class HistogramEqualizationTestSuite(unittest.TestSuite):
    pass


class GradientTestSuite(unittest.TestSuite):
    pass



class TestImageFunctions(unittest.TestCase):
    def test_get_image_data(self):
        original = get_image_data("testimg/preprocessed/tree.jpg", "L")
        copied = get_image_data("testimg/processed/copy/tree.jpg", "L")
        self.assertTrue((original == copied).all())

    def test_get_new_image(self):
        pass


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


class TestCDFFunctions(unittest.TestCase):
    def test_get_cdf(self):
        pass

    def test_get_rgb_cdf(self):
        pass


class TestEqualizeFunctions(unittest.TestCase):
    def test_equalize(self):
        pass

    def test_equalize_rgb(self):
        pass


class TestHistogramTransformFunctions(unittest.TestCase):
    def test_transform_grayscale(self):
        pass

    def test_transform_rgb(self):
        pass

    def test_transform_hsv(self):
        pass

    def test_transform_split(self):
        pass


class TestConvenienceFunctions(unittest.TestCase):
    def test_rgb_split(self):
        pass

    def test_equalize_multichannel(self):
        pass

    def test_equalize_singlechannel(self):
        pass


class TestGradientFunctions(unittest.TestCase):
    def test_vertical_gradient(self):
        pass
