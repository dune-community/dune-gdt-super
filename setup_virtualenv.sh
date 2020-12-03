virtualenv -p python3.8 venv
source venv/bin/activate
python -m pip install numpy scipy cython mpi4py
cd pymor
pip install -e .
cd ..
