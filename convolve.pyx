# convolve.pyx

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int 
ctypedef np.int_t DTYPE_t
DTYPE8 = np.uint8
ctypedef np.uint8_t DTYPE8_t

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve(np.ndarray[DTYPE8_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
    cdef int x, y, s, t, v, w, s_from, s_to, t_from, t_to
    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2 * smid
    cdef int ymax = wmax + 2 * tmid
    
    cdef DTYPE8_t value
    cdef np.ndarray[DTYPE8_t, ndim=2] h = np.zeros([xmax, ymax], dtype=DTYPE8)
    
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
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h
