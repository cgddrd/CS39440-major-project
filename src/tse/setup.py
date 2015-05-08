import numpy as np

from pip.req import parse_requirements
from pip.download import PipSession

from setuptools import setup, Extension

# Required in order to build the Cython C extensions.
from Cython.Distutils import build_ext

# Read from 'requirements.txt' file. Modified from: http://stackoverflow.com/a/16624700
# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("./requirements.txt", session=PipSession())

# reqs is a list of requirements
reqs = [str(ir.req) for ir in install_reqs]

# Define the Cython extensions that should be compiled prior to installing the library.
ext = Extension("tse_compiled.tse_c_imageutils", ["tse_compiled/tse_c_imageutils.pyx"], include_dirs=[np.get_include()])

setup(
    name='tse',
    version='3.1.1',
    zip_safe=False,
    packages=['tse', 'tse_compiled', 'tests'],
    install_requires=reqs,
    url='https://github.com/cgddrd/CS39440-major-project',
    author='Connor Goddard',
    author_email='connorlukegoddard@gmail.com',
    description='Terrain Shape Estimation library providing core shared functionality for conduting experiments regarding appearance-based template matching.',
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext]
)
