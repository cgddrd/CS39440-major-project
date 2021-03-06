# Install dependencies using 'pip' - (Note: setuptools seems to break if we try and install dependencies in 'setup.py'.)
pip install -r requirements.txt

# Compile/build the Cython 'C' extensions.
python setup.py build_ext --inplace

# Run the 'setup.py' script to install dependencies etc.
python setup.py install

# Run the automated unti tests with coverage.
nosetests -w tests/ --with-coverage --cover-package=tse --cover-package=tse_compiled  --cover-erase
