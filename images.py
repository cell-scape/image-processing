import sys
from itertools import chain

import numpy as np
from PIL import Image

BITS = 256

def get_image_data(filename, mode="L"):
    return np.asarray(Image.open(filename).convert(mode))

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
    return (red, green, blue)

def get_hsv_histogram(imgdata):
    value = [0] * BITS
    for i in chain.from_iterable(imgdata):
        value[i[2]] += 1
    return value

def get_cdf(histogram):
    cdf = [0] * BITS
    cdf[0] = histogram[0]
    for i in range(1, BITS):
        cdf[i] = cdf[i-1] + histogram[i]
    return cdf

def get_rgb_cdf(red, green, blue):
    return (get_cdf(red), get_cdf(green), get_cdf(blue))

def equalize(cdf):
    nzmin = min(filter(lambda x: x > 0, cdf))
    return [round(((cdf[i] - nzmin) / (cdf[-1] - nzmin)) * (BITS - 1)) for i in range(BITS)]

def equalize_rgb(red, green, blue):
    return (equalize(red), equalize(green), equalize(blue))

def transform_grayscale(equalized, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = equalized[indata[i, j]]
    return outdata

def transform_rgb(red, green, blue, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [red[indata[i, j, 0]], green[indata[i, j, 1]], blue[indata[i, j, 2]]]
    return outdata

def transform_red(red, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [red[indata[i, j, 0]], 0, 0]
    return outdata

def transform_green(green, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [0, green[indata[i, j, 1]], 0]
    return outdata

def transform_blue(blue, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [0, 0, blue[indata[i, j, 2]]]
    return outdata

def transform_hsv(equalized, indata):
    outdata = np.zeros_like(indata)
    for i in range(indata.shape[0]):
        for j in range(indata.shape[1]):
            outdata[i, j] = [indata[i, j, 0], indata[i, j, 1], equalized[indata[i, j, 2]]]
    return outdata

def get_new_image(outdata, mode="L"):
    return Image.fromarray(outdata, mode)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Usage: python images.py image.format [L | RGB [SPLIT]]")

    fname = sys.argv[1]
    mode = sys.argv[2].upper()    
    indata = get_image_data(fname, mode)
    if mode == "L" or mode == "P":
        hist = get_grayscale_histogram(indata)
        cdf = get_cdf(hist)
        equalized = equalize(cdf)
        outdata = transform_grayscale(equalized, indata)
        new_image = get_new_image(outdata, mode)
        new_image.show()
    elif mode == "RGB" and len(sys.argv) < 4:
        red, green, blue = get_rgb_histogram(indata)
        rcdf, gcdf, bcdf = get_rgb_cdf(red, green, blue)
        req, geq, beq = equalize_rgb(rcdf, gcdf, bcdf)
        outdata = transform_rgb(req, geq, beq, indata)
        new_image = get_new_image(outdata, mode)
        new_image.show()
    elif mode == "HSV" and len(sys.argv) < 4:
        hist = get_hsv_histogram(indata)
        cdf = get_cdf(hist)
        equalized = equalize(cdf)
        outdata = transform_hsv(equalized, indata)
        new_image = get_new_image(outdata, mode)
        new_image.show()
    elif mode == "HSV" and sys.argv[3] == "all":
        h, s, v = get_rgb_histogram(indata)
        hcdf, scdf, vcdf = get_rgb_cdf(h, s, v)
        heq, seq, veq = equalize_rgb(h, s, v)
        outdata = transform_rgb(heq, seq, veq, indata)
        new_image = get_new_image(outdata, mode)
        new_image.show()
    elif mode == "HSV" and sys.argv[3] == "value":
        v = get_hsv_histogram(indata)
        vcdf = get_cdf(v)
        veq = equalize(vcdf)
        outdata = transform_blue(veq, indata)
        new_image = get_new_image(outdata, "HSV")
        new_image.show()
    elif mode == "RGB" and sys.argv[3] == "split":
        red, green, blue = get_rgb_histogram(indata)
        rcdf, gcdf, bcdf = get_rgb_cdf(red, green, blue)
        req, geq, beq = equalize_rgb(rcdf, gcdf, bcdf)
        outred = transform_red(req, indata)
        outgreen = transform_green(geq, indata)
        outblue = transform_blue(beq, indata)
        new_red = get_new_image(outred, "RGB")
        new_green = get_new_image(outgreen, "RGB")
        new_blue = get_new_image(outblue, "RGB")
        new_red.show()
        new_green.show()
        new_blue.show()
    else:
        sys.exit("Choose mode L or RGB")
sys.exit(0)
    