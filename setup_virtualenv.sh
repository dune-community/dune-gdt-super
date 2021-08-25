virtualenv -p python3 venv
source venv/bin/activate
python -m pip --no-cache-dir install numpy scipy cython mpi4py rich
cd pymor
pip install -e .
cd ..
