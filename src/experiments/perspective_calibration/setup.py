try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='perspective_calibration',
    version='1.0.0',
    packages=['', 'tests'],
    install_requires=[
        'matplotlib'
    ],
    url='https://github.com/cgddrd/CS39440-major-project',
    license='',
    author='Connor Goddard',
    author_email='connorlukegoddard@gmail.com',
    description=''
)
