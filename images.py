import sys
import os.path
import argparse
from itertools import chain

from convolve import convolve

import numpy as np
from PIL import Image

# Constants: Bit depth and gradient operators
BITS = 256 
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

def naive_convolve(imgdata, kernel):
    vmax, wmax = imgdata.shape
    smax, tmax = kernel.shape
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2 * smid
    ymax = wmax + 2 * tmid
    outdata = np.zeros([xmax, ymax], dtype=np.uint8)
    for x in range(xmax):
        for y in range(ymax):
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += kernel[smid - s, tmid - t] * imgdata[v, w]
            outdata[x, y] = value
    return outdata

def sobel(indata):
    """Pass to gradient_images(): Convolves Sobel operator with image data"""
    sx = naive_convolve(indata, SOBEL_GX)
    sy = naive_convolve(indata, SOBEL_GY)
    return sx, sy, np.hypot(sx, sy)

def prewitt(indata):
    """Pass to gradient_images(): Convolves Prewitt operator with image data"""
    px = convolve(indata, PREWITT_GX)
    py = convolve(indata, PREWITT_GY)
    return px, py, np.hypot(px, py)

def scharr(indata):
    """Pass to gradient_images(): Convolves common Scharr operator with image data"""
    scx = convolve(indata, SCHARR_GX)
    scy = convolve(indata, SCHARR_GY)
    return scx, scy, np.hypot(scx, scy)

def optimal(indata):
    """Pass to gradient_images(): Convolves optimal Scharr operator with images data"""
    ox = convolve(indata, OPTIMAL_GX)
    oy = convolve(indata, OPTIMAL_GY)
    return ox, oy, np.hypot(ox, oy)

def gradient_images(imgdata, kernel):
    """Returns tuple of x, y, and xy gradient images. Takes image data and a kernel"""
    x, y, sobel = kernel(imgdata)
    return (get_new_image(x, "L"),
            get_new_image(y, "L"),
            get_new_image(sobel, "L"))


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
    parser.add_argument("-i",
                        help="Show initial images",
                        action="store_true",
                        dest="initial",
                        required=False)
    parser.add_argument("-o",
                        help="Select operator for gradients",
                        choices=["sobel", "prewitt", "scharr", "optimal"],
                        dest="operator",
                        default="sobel",
                        metavar="OPERATOR",
                        required=False)

    optiongroup.add_argument("-s", "--split",
                            help="split image into separate R, G, and B channel images",
                            action="store_true",)
    optiongroup.add_argument("-a", "--all",
                            help="hist. eq. on all HSV channels, or all gradient types",
                            action="store_true")
    optiongroup.add_argument("-x",
                             help="Gradient wrt X",
                             dest="x",
                             action="store_true")
    optiongroup.add_argument("-y",
                             help="Gradient wrt Y",
                             dest="y",
                             action="store_true")
    optiongroup.add_argument("-xy",
                             help="XY gradient: sqrt(x^2 + y^2)",
                             dest="xy",
                             action="store_true")
    return parser

if __name__ == '__main__':
    args = setup_argparse().parse_args()

    filename = os.path.abspath(args.filename)
    if not os.path.exists(filename):
        sys.exit(f"{os.path.basename(filename)} not found")

    mode = args.mode.upper()
    grad = False
    if mode == "GRADIENT":
        grad = True
        mode = "L"

    indata = get_image_data(filename, mode)
    if args.initial:
        get_new_image(indata, mode).show()
        if args.split:
            rgb_split_preprocess(indata)

    if grad:
        x, y, xy = gradient_images(indata, eval(args.operator))
        if args.x:
            x.show()
        elif args.y:
            y.show()
        elif args.xy:
            xy.show()
        else:
            x.show()
            y.show()
            xy.show()

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
