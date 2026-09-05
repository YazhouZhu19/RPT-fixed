from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup


extensions = [
    Extension("felzenszwalb_3d_cy", ["felzenszwalb_3d_cy.pyx"], include_dirs=[numpy.get_include()]),
    Extension("_ccomp", ["_ccomp.pyx"], include_dirs=[numpy.get_include()]),
]
setup(
    name='rpt-supervoxels',
    ext_modules=cythonize(extensions)
)
