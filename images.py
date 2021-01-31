#! ./bin/python3
# -*- encoding: utf-8 -*-

import sys
import os.path
import argparse
from itertools import chain

import numpy as np
from PIL import Image

BITS = 256


# Image IO

def get_image_data(filename, mode):
    return np.asarray(Image.open(filename).convert(mode))


def get_new_image(outdata, mode):
    return Image.fromarray(outdata, mode)


# Histogram Functions

def get_grayscale_histogram(imgdata):
    histogram = [0] * BITS
    for pixel in imgdata.flatten():
        histogram[pixel] += 1
    return histogram

def get_rgb_histogram(imgdata):
    red = [0] * BITS
    green = [0] * BITS
    blue = [0] * BITS
    for color in chain.from_iterable(imgdata):
        red[color[0]] += 1
        green[color[1]] += 1
        blue[color[2]] += 1
    return red, green, blue

def get_hsv_histogram(imgdata):
    value = [0] * BITS
    for i in chain.from_iterable(imgdata):
        value[i[2]] += 1
    return value


# Cumulative Distribution Functions

def get_cdf(histogram):
    cdf = [0] * BITS
    cdf[0] = histogram[0]
    for i in range(1, BITS):
        cdf[i] = cdf[i-1] + histogram[i]
    return cdf

def get_rgb_cdf(rgb):
    return get_cdf(rgb[0]), get_cdf(rgb[1]), get_cdf(rgb[2])


# Equalization Functions

def equalize(cdf):
    nzmin = min(filter(lambda x: x > 0, cdf))
    if cdf[-1] - nzmin == 0:
        sys.exit("Image cannot be equalized in all channels (div by zero)")
    return [round(((cdf[i] - nzmin) / (cdf[-1] - nzmin)) * (BITS - 1)) for i in range(BITS)]

def equalize_rgb(rgbcdf):
    return equalize(rgbcdf[0]), equalize(rgbcdf[1]), equalize(rgbcdf[2])


# Image Data Transformation Functions

def transform_grayscale(equalized, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = equalized[indata[i, j]]
    return outdata

def transform_rgb(rgb, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [rgb[0][indata[i, j, 0]],
                             rgb[1][indata[i, j, 1]],
                             rgb[2][indata[i, j, 2]]]
    return outdata

def transform_hsv(equalized, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [indata[i, j, 0],
                             indata[i, j, 1],
                             equalized[indata[i, j, 2]]]
    return outdata

def transform_split(rgb, indata):
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



# Convenience Functions

def rgb_split_preprocess(indata):
    hist = get_grayscale_histogram(indata)
    outdata = transform_split((hist, hist, hist), indata)
    get_new_image(outdata[0], "RGB").show()
    get_new_image(outdata[1], "RGB").show()
    get_new_image(outdata[2], "RGB").show()
    return
    

def rgb_split(rgbeq, indata):
    outdata = transform_split(rgbeq, indata)
    return  (get_new_image(outdata[0], "RGB"),
             get_new_image(outdata[1], "RGB"),
             get_new_image(outdata[2], "RGB"))

def equalize_multichannel(indata):
    return equalize_rgb(get_rgb_cdf(get_rgb_histogram(indata)))

def equalize_singlechannel(indata):
    return equalize(get_cdf(get_grayscale_histogram(indata)))


# CLI Interface : Eventually Separate Out
# Separate Histogram Equalization module from Gradient Module also

parser = argparse.ArgumentParser(
    description="Process image file using Histogram Equalization or X-Y Gradient"
)
optiongroup = parser.add_mutually_exclusive_group()
parser.add_argument("filename", help="The path to the image file")
parser.add_argument("mode",
                    help="The image file mode: " + 
                    "l = Grayscale, p = Palette, rgb = True Color, hsv = Hue-Saturation-Value",
                    choices=["l", "p", "rgb", "hsv"],
                    default="l")
parser.add_argument("-i", "--initial",
                    help="Show original images",
                    action="store_true",
                    dest="initial",
                    required=False)
optiongroup.add_argument("-s", "--split",
                         help="split image into separate R, G, and B channel images",
                         action="store_true",)
optiongroup.add_argument("-a", "--all",
                         help="perform histogram equalization on all HSV channels",
                         action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    
    filename = os.path.abspath(args.filename)
    if not os.path.exists(filename):
        sys.exit(f"{os.path.basename(filename)} not found")
        
    mode = args.mode.upper()
    indata = get_image_data(filename, mode)

    if args.initial:
        get_new_image(indata, mode).show()
        if args.split:
            rgb_split_preprocess(indata)
    
    if mode == "L" or mode == "P":
        eq = equalize_singlechannel(indata)
        if args.split:
            indata = get_image_data(filename, "RGB")
            redimg, greenimg, blueimg = rgb_split((eq, eq, eq), indata)
            redimg.show()
            greenimg.show()
            blueimg.show()
        else:
            outdata = transform_grayscale(eq, indata)
            grayscaleimg = get_new_image(outdata, mode)
            grayscaleimg.show()
    elif mode == "RGB":
        eq = equalize_multichannel(indata)
        if args.split:
            redimg, greenimg, blueimg = rgb_split(eq, indata)
            redimg.show()
            greenimg.show()
            blueimg.show()
        else:
            outdata = transform_rgb(eq, indata)
            rgbimg = get_new_image(outdata, mode)
            rgbimg.show()
    elif mode == "HSV":
        if args.all:
            outdata = transform_rgb(equalize_multichannel(indata), indata)
            allhsvimg = get_new_image(outdata, mode)
            allhsvimg.show()
        else:
            hist = get_hsv_histogram(indata)
            outdata = transform_hsv(equalize(get_cdf(hist)), indata)
            hsvimg = get_new_image(outdata, mode)
            hsvimg.show()
    else:
        sys.exit(-1)
    sys.exit(0)
