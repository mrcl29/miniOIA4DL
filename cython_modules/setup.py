from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cython_modules/*.pyx", language_level=3),
    include_dirs=[numpy.get_include()], #indicar al compilador que coja los ficheros .h internos de numpy
) 