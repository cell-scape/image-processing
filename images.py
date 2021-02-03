import sys
import os.path
import argparse
from math import sqrt
from itertools import chain

import numpy as np
from PIL import Image

from convolve import convolve

# Constants: Bit depth and gradient operators
BITS = 256
BASIC_X = np.array([[1],[-1]])
BASIC_Y = np.array([[-1], [1]])
ROBERTS = np.array([[1, 0], [0, -1]])
PREWITT_GX = np.array([[1, 0 , -1], [1, 0, -1], [1, 0, -1]])
PREWITT_GY = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
SOBEL_GX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
SOBEL_GY = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
SCHARR_GX = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
SCHARR_GY = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
OPTIMAL_GX = np.array([[47, 162, 47], [0, 0, 0], [-47, -162, -47]])
OPTIMAL_GY = np.array([[47, 0, -47], [162, 0, -162], [47, 0, -47]])


# Image IO

def get_image_data(filename, mode, dtype=np.uint8):
    """Return numpy array from PIL Image object"""
    return np.asarray(Image.open(filename).convert(mode), dtype=dtype)


def get_new_image(outdata, mode):
    """Return PIL Image object from numpy array"""
    return Image.fromarray(outdata, mode)


# Histogram Functions

def get_grayscale_histogram(imgdata):
    """Return array of counts of luminosity levels"""
    histogram = [0] * BITS
    for pixel in imgdata.flatten():
        histogram[pixel] += 1
    return histogram

def get_rgb_histogram(imgdata):
    """Returns tuple of arrays of counts for R, G, B channels"""
    red = [0] * BITS
    green = [0] * BITS
    blue = [0] * BITS
    for color in chain.from_iterable(imgdata):
        red[color[0]] += 1
        green[color[1]] += 1
        blue[color[2]] += 1
    return red, green, blue

def get_hsv_histogram(imgdata):
    """Returns array of counts of value channel"""
    value = [0] * BITS
    for i in chain.from_iterable(imgdata):
        value[i[2]] += 1
    return value


# Cumulative Distribution Functions

def get_cdf(histogram):
    """Return cumulative distribution function of histogram array"""
    cdf = [0] * BITS
    cdf[0] = histogram[0]
    for i in range(1, BITS):
        cdf[i] = cdf[i-1] + histogram[i]
    return cdf

def get_rgb_cdf(rgb):
    """Returns one cumulative distribution function for each channel"""
    return get_cdf(rgb[0]), get_cdf(rgb[1]), get_cdf(rgb[2])


# Equalization Functions

def equalize(cdf):
    """Equalize histogram based on cumulative distribution function"""
    nzmin = min(filter(lambda x: x > 0, cdf))
    if cdf[-1] - nzmin == 0:
        sys.exit("Image cannot be equalized in all channels (div by zero)")
    return [round(((cdf[i] - nzmin) / (cdf[-1] - nzmin)) * (BITS - 1)) for i in range(BITS)]

def equalize_rgb(rgbcdf):
    return equalize(rgbcdf[0]), equalize(rgbcdf[1]), equalize(rgbcdf[2])


# Image Data Transformation Functions

def transform_grayscale(equalized, indata):
    """Return numpy array of luminosity levels mapped from equalized histogram"""
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = equalized[indata[i, j]]
    return outdata

def transform_rgb(rgb, indata):
    """Return numpy array of R, G, B channels mapped from each equalized histogram"""
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [rgb[0][indata[i, j, 0]],
                             rgb[1][indata[i, j, 1]],
                             rgb[2][indata[i, j, 2]]]
    return outdata

def transform_hsv(equalized, indata):
    """Return numpy array of H, S, V channels, H, S unchanged, V equalized"""
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [indata[i, j, 0],
                             indata[i, j, 1],
                             equalized[indata[i, j, 2]]]
    return outdata

def transform_split(rgb, indata):
    """Return tuple of numpy arrays of separate equalized R, G, B channels"""
    redout = np.zeros_like(indata)
    greenout = np.zeros_like(indata)
    blueout = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            redout[i, j] = [rgb[0][indata[i, j, 0]], 0, 0]
            greenout[i, j] = [0, rgb[1][indata[i, j, 1]], 0]
            blueout[i, j] = [0, 0, rgb[2][indata[i, j, 2]]]
    return redout, greenout, blueout


# Gradient Functions

def native_convolve(imgdata, kernel, threshold, normalize):
    imgx, imgy = imgdata.shape
    kx, ky = kernel.shape
    kxmid = kx // 2
    kymid = ky // 2
    outx = imgx + 2 * kxmid
    outy = imgy + 2 * kymid
    outdata = np.zeros([outx, outy], dtype=np.uint8)
    for x in range(outx):
        for y in range(outy):
            kx_from = max(kxmid - x, -kxmid)
            kx_to = min((outx - x) - kxmid, kxmid + 1)
            ky_from = max(kymid - y, -kymid)
            ky_to = min((outy - y) - kymid, kymid + 1)
            value = 0
            for s in range(kx_from, kx_to):
                for t in range(ky_from, ky_to):
                    v = x - kxmid + s
                    w = y - kymid + t
                    value += np.absolute(kernel[kxmid - s, kymid - t] * imgdata[v, w])
                    if value < threshold:
                        value = imgdata[v, w]
            outdata[x, y] = value
    return outdata

def basic(indata, threshold=0, norm=0):
    x = convolve(indata, BASIC_X, threshold, norm)
    y = convolve(indata, BASIC_Y, threshold, norm)
    return x, y

def roberts(indata, threshold=0, norm=0):
    get_new_image(convolve(indata, ROBERTS, threshold, norm), "L").show()
    sys.exit(0)

def sobel(indata, threshold=0, norm=0):
    """Pass to gradient_images(): Convolves Sobel operator with image data"""
    sx = convolve(indata, SOBEL_GX, threshold, norm)
    sy = convolve(indata, SOBEL_GY, threshold, norm)
    return sx, sy

def prewitt(indata, threshold=0, norm=0):
    """Pass to gradient_images(): Convolves Prewitt operator with image data"""
    px = convolve(indata, PREWITT_GX, threshold, norm)
    py = convolve(indata, PREWITT_GY, threshold, norm)
    return px, py

def scharr(indata, threshold=0, norm=0):
    """Pass to gradient_images(): Convolves common Scharr operator with image data"""
    scx = convolve(indata, SCHARR_GX, threshold, norm)
    scy = convolve(indata, SCHARR_GY, threshold, norm)
    return scx, scy

def optimal(indata, threshold=0, norm=0):
    """Pass to gradient_images(): Convolves optimal Scharr operator with images data"""
    ox = convolve(indata, OPTIMAL_GX, threshold, norm)
    oy = convolve(indata, OPTIMAL_GY, threshold, norm)
    return ox, oy

def gradient_images(imgdata, kernel, threshold=0, norm=0, m=False):
    """Returns tuple of x, y, and mag gradient images. Takes image data and a kernel"""
    if m:
        x, y = kernel(imgdata, 0, norm)
        mag = gmag(x, y, threshold)
    else:
        x, y = kernel(imgdata, threshold, norm)
        mag = gmag(x, y, threshold)
    return (get_new_image(x, "L"),
            get_new_image(y, "L"),
            get_new_image(mag[0], "L"),
            get_new_image(mag[1], "L"))

def gmag(gx, gy, threshold=0):
    row, col = gx.shape
    out = np.zeros_like(gx)
    timg = np.zeros_like(gx)
    for i in range(row):
        for j in range(col):
            val = sqrt(gx[i, j]**2 + gy[i, j]**2)
            if val > threshold:
                out[i, j] = val
                timg[i, j] = 255
            else:
                out[i, j] = 0
                timg[i, j] = 0
    return out, timg


# Convenience Functions

def rgb_split_preprocess(indata):
    """Show individual R, G, B channels before any processing"""
    hist = get_grayscale_histogram(indata)
    outdata = transform_split((hist, hist, hist), indata)
    get_new_image(outdata[0], "RGB").show()
    get_new_image(outdata[1], "RGB").show()
    get_new_image(outdata[2], "RGB").show()


def rgb_split(rgbeq, indata):
    """Show transformed R, G, B channels"""
    outdata = transform_split(rgbeq, indata)
    get_new_image(outdata[0], "RGB").show()
    get_new_image(outdata[1], "RGB").show()
    get_new_image(outdata[2], "RGB").show()


def equalize_multichannel(indata):
    """Applies histogram | cdf | equalize for three channel images"""
    return equalize_rgb(get_rgb_cdf(get_rgb_histogram(indata)))

def equalize_singlechannel(indata):
    """Applies histogram | cdf | equalize for grayscale images"""
    return equalize(get_cdf(get_grayscale_histogram(indata)))


# Command line interface

def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Process image file using Histogram Equalization or X-Y Gradient"
    )
    optiongroup = parser.add_mutually_exclusive_group()
    
    parser.add_argument("filename", help="The path to the image file")
    parser.add_argument("mode",
                        help="The image file mode: l, p, rgb, hsv, gradient",
                        choices=["l", "p", "rgb", "hsv", "gradient"],
                        default="l")
    parser.add_argument("-i", "--initial",
                        help="Show initial images",
                        action="store_true",
                        dest="initial",
                        required=False)
    parser.add_argument("-o", "--operator",
                        help="Select operator for gradients: sobel, prewitt, scharr, optimal",
                        choices=["sobel", "prewitt", "scharr", "optimal", "basic", "roberts"],
                        dest="operator",
                        default="sobel",
                        metavar="OPERATOR",
                        required=False)
    parser.add_argument("-t", "--threshold",
                        help="Select value from 0-255 as gradient threshold",
                        dest="threshold",
                        metavar="N",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("-m", "-magnitude-threshold",
                        dest="m",
                        help="Apply threshold value only to Magnitude",
                        action="store_true",
                        required=False)
    parser.add_argument("-n", "--normalize",
                        help="Locally Normalize Gradient",
                        dest="norm",
                        action="store_true",
                        required=False)

    optiongroup.add_argument("-s", "--split",
                            help="split image into separate R, G, and B channel images",
                            action="store_true",)
    optiongroup.add_argument("-a", "--all",
                            help="hist. eq. on all HSV channels, or all gradient types",
                            action="store_true")
    optiongroup.add_argument("-x", "--X",
                             help="Gradient wrt X",
                             dest="x",
                             action="store_true")
    optiongroup.add_argument("-y", "--Y",
                             help="Gradient wrt Y",
                             dest="y",
                             action="store_true")
    optiongroup.add_argument("-mag", "--magnitude",
                             help="gradient magnitude: sqrt(x^2 + y^2)",
                             dest="mag",
                             action="store_true")
    return parser

if __name__ == '__main__':
    args = setup_argparse().parse_args()

    filename = os.path.abspath(args.filename)
    if not os.path.exists(filename):
        sys.exit(f"{os.path.basename(filename)} not found")

    mode = args.mode.upper()
    grad = False
    m = False
    norm = 0
    if mode == "GRADIENT":
        grad = True
        mode = "L"

    indata = get_image_data(filename, mode)
    if args.initial:
        get_new_image(indata, mode).show()
        if args.split:
            rgb_split_preprocess(indata)

    if grad:
        if args.norm:
            norm = 1
            
        if args.m:
            m = True
            
        if args.threshold < 0 or args.threshold > 255:
            sys.exit("Threshold value must be 0-255")
        x, y, mag, thr = gradient_images(indata, eval(args.operator), args.threshold, norm, m=False)
        if args.x:
            x.show()
        elif args.y:
            y.show()
        elif args.mag:
            mag.show()
            thr.show()
        else:
            x.show()
            y.show()
            mag.show()
            thr.show()

    elif mode == "L" or mode == "P":
        eq = equalize_singlechannel(indata)
        if args.split:
            indata = get_image_data(filename, "RGB")
            rgb_split((eq, eq, eq), indata)
        else:
            outdata = transform_grayscale(eq, indata)
            get_new_image(outdata, mode).show()

    elif mode == "RGB":
        eq = equalize_multichannel(indata)
        if args.split:
            rgb_split(eq, indata)
        else:
            outdata = transform_rgb(eq, indata)
            get_new_image(outdata, mode).show()

    elif mode == "HSV":
        if args.all:
            outdata = transform_rgb(equalize_multichannel(indata), indata)
            get_new_image(outdata, mode).show()
        else:
            hist = get_hsv_histogram(indata)
            outdata = transform_hsv(equalize(get_cdf(hist)), indata)
            get_new_image(outdata, mode).show()

    else:
        sys.exit(-1)

    sys.exit(0)
