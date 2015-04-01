import numpy as np
import sys

from pip.req import parse_requirements
from pip.download import PipSession

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

# from distutils.extension import Extension
# Read from 'requirements.txt' file. Modified from: http://stackoverflow.com/a/16624700

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("./requirements.txt", session=PipSession())

# reqs is a list of requirements
reqs = [str(ir.req) for ir in install_reqs]

# setuptools DWIM monkey-patch madness
# http://mail.python.org/pipermail/distutils-sig/2007-September/thread.html#8204
# if 'setuptools.extension' in sys.modules:
#     m = sys.modules['setuptools.extension']
#     m.Extension.__dict__ = m._Extension.__dict__

# To install module in package folder, need to use package '.' notation in extension name.
# See: https://github.com/uci-cbcl/tree-hmm/issues/2

ext = Extension("tse_compiled.tse_c_imageutils", ["tse_compiled/tse_c_imageutils.pyx"], include_dirs=[np.get_include()])

setup(
    name='template_matching_scaling',
    version='2.1.0',
    zip_safe=False,
    packages=['tse', 'tse_compiled', 'tests'],
    # setup_requires=['cython', 'setuptools_cython'],
    install_requires=reqs,
    url='https://github.com/cgddrd/CS39440-major-project',
    author='Connor Goddard',
    author_email='connorlukegoddard@gmail.com',
    description='',
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext]
)

