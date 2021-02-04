# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel cimport prange, parallel
cimport cython
cimport openmp
from libc.math cimport sqrt, round

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t
DTYPE8 = np.uint8
ctypedef np.uint8_t DTYPE8_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convolve(np.ndarray[DTYPE8_t, ndim=2] img, np.ndarray[DTYPE_t, ndim=2] window, DTYPE8_t thr, bint norm):
    cdef int x, y, s, t, v, w, i, wx_from, wx_to, wy_from, wy_to, lsum, sos
    cdef unsigned char ctr, idx, sd, mean, sqd
    cdef float var
    cdef int imgx = img.shape[0]
    cdef int imgy = img.shape[1]
    cdef int winx = window.shape[0]
    cdef int winy = window.shape[1]
    cdef int wxmid = winx // 2
    cdef int wymid = winy // 2
    cdef int outx = imgx + 2 * wxmid
    cdef int outy = imgy + 2 * wymid

    cdef np.ndarray[DTYPE8_t, ndim=1] neighborhood = np.zeros([winx*winy], dtype=DTYPE8)
    cdef DTYPE8_t value
    cdef np.ndarray[DTYPE8_t, ndim=2] out = np.zeros([outx, outy], dtype=DTYPE8)
    with nogil, parallel(num_threads=8):
        for x in prange(outx, schedule='guided'):
            for y in range(outy):
                wx_from = max(wxmid - x, -wxmid)
                wx_to = min((outx - x) - wxmid, wxmid + 1)
                wy_from = max(wymid - y, -wymid)
                wy_to = min((outy - y) - wymid, wymid + 1)
                value = 0
                ctr = 0
                lsum = 0
                for s in range(wx_from, wx_to):
                    for t in range(wy_from, wy_to):
                        v = x - wxmid + s
                        w = y - wymid + t
                        if norm:
                            neighborhood[ctr] = img[v, w]
                            ctr = ctr + 1
                            lsum = lsum + img[v, w]
                        value = value + window[wxmid - s, wymid - t] * img[v, w]
                        # Abs
                        if value < 0:
                            value = -value
                        # Threshold
                        if thr > 0 and value < thr:
                            value = img[v, w]
                # Locally Normalize Value within window
                if norm:
                    idx = (wx_to - wx_from) * (wy_to - wy_from)
                    mean = lsum / idx
                    sd = 0
                    sos = 0
                    for i in range(idx):
                        if neighborhood[i] == 0:
                            continue
                        sqd = (neighborhood[i] - mean) * (neighborhood[i] - mean)
                        sos = sos + sqd
                        neighborhood[i] = 0
                    var = sos / idx
                    sd = <unsigned char>round(sqrt(var))
                    if mean > 0:
                        value = <unsigned char>((value - sd) // mean)
                out[x, y] = value
    return out
