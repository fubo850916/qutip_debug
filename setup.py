from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("cy_ode", ["cy_ode.pyx"], 
    define_macros=[('CYTHON_TRACE_NOGIL', '1')],
    language='c++')
]

INCLUDE_DIRS = [np.get_include(),'/usr/include'] if np is not None else []

setup(
    ext_modules = cythonize(extensions),
    include_dirs = INCLUDE_DIRS
)
