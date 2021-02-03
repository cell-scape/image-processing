from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("convolve", ["convolve.pyx"],
                         include_dirs=[np.get_include()],
                         extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp', '-lm'],
                         extra_link_args=['-fopenmp'],
                         libraries=["m"])]

setup(
    name="2D Convolution",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
