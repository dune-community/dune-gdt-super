virtualenv -p python3 venv
source venv/bin/activate
python -m pip install numpy scipy cython mpi4py
cd pymor
python setup.py install
cd ..
