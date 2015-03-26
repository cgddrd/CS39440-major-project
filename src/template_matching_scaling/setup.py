from pip.req import parse_requirements
from pip.download import PipSession

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Read from 'requirements.txt' file. Modified from: http://stackoverflow.com/a/16624700

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("./requirements.txt", session=PipSession())

# reqs is a list of requirements
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='perspective_calibration',
    version='1.0.1',
    packages=['', 'tests'],
    install_requires=reqs,
    url='https://github.com/cgddrd/CS39440-major-project',
    author='Connor Goddard',
    author_email='connorlukegoddard@gmail.com',
    description=''
)
