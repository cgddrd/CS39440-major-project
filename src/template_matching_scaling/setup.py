import numpy
import sys

from pip.req import parse_requirements
from pip.download import PipSession

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

# Read from 'requirements.txt' file. Modified from: http://stackoverflow.com/a/16624700

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("./requirements.txt", session=PipSession())

# reqs is a list of requirements
reqs = [str(ir.req) for ir in install_reqs]


if 'setuptools.extension' in sys.modules:
    m = sys.modules['setuptools.extension']
    m.Extension.__dict__ = m._Extension.__dict__


ext = Extension("test2", ["tse_c_imageutils.pyx"], include_dirs=[numpy.get_include()])


setup(
    name='template_matching_scaling',
    version='1.0.0',
    packages=['', 'tests', 'tse'],
    install_requires=reqs,
    url='https://github.com/cgddrd/CS39440-major-project',
    author='Connor Goddard',
    author_email='connorlukegoddard@gmail.com',
    description='',
    setup_requires=['setuptools_cython', 'numpy'],
    ext_modules=[ext]
)

