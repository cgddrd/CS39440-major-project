
# Compile/build the Cython 'C' extensions.
# python setup.py build_ext --inplace

# Run the 'setup.py' script to install dependencies etc.
python setup.py install

# Actually install the Python module.
pip install .

ls -l

cd tse_compiled

ls -l

cd ..

# Run the automated unti tests with coverage.
nosetests -w tests/ --with-coverage
