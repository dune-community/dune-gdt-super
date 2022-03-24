```
# This file is part of the dune-gdt-super project:
#   https://github.com/dune-community/dune-gdt-super
# The copyright lies with the authors of this file (see below).
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```
Dieses Repository enthält den Code für die numerischen Experimente zum HAPOD2 Paper.

# Installieren und Ausführen auf normalem Rechner oder Jaina/Jacen/Syal/Myri etc:

## Installieren

```bash
git clone git@zivgitlab.uni-muenster.de:ag-ohlberger/dune-community/dune-gdt-super.git
cd dune-gdt-super
git checkout hapod2
git submodule update --init --recursive
CC=gcc ./bin/build_external_libraries.py
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release.ninja all
source build/gcc-release/dune-python-env/bin/activate
python3 -m pip install 
python -m pip --no-cache-dir install numpy scipy cython mpi4py rich neovim
cd pymor
pip install -e .
cd ..
cd build/gcc-release/dune-xt
ninja && ninja bindings
cd ../dune-gdt
ninja && ninja bindings
./add_libhapodgdt_symlinks.sh
```

## Ausführen

```bash
source build/gcc-release/dune-python-env/bin/activate
mpiexec -n 32 python3 cellmodel_hapod_deim.py testcase t_end dt grid_size_x grid_size_y pfield_tol ofield_tol stokes_tol pfield_deim_tol ofield_deim_tol stokes_deim_tol calculate_errors parameter_sampling_type pod_method visualize subsampling include_newton_stages
```
Die Kommandozeilenargumente sind:
- testcase: Testcase, relevant sind nur die Optionen cell_isolation_experiment (das Hauptexperiment aus dem Paper) oder single_cell (einfache Runde Zelle, einmal im Paper benutzt zum Testen der Solver)
- t_end: Endzeitpunkt
- dt: Zeitschrittweite
- grid_size_x: Gittergröße in x-Richtung
- grid_siye_y: Gittergröße in y-Richtung
- pfield_tol: Prescribed mean l2 error für die Phasenfeld HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- ofield_tol: Prescribed mean l2 error für die Orientierungsfeld HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- stokes_tol: Prescribed mean l2 error für die Stokes HAPOD (wenn 0, dann keine Reduktion in dieser Variable)
- pfield_deim_tol: Prescribed mean l2 error für die Phasenfeld HAPOD für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- ofield_deim_tol: Prescribed mean l2 error für die Orientierungsfeld HAPOD  für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- stokes_deim_tol: Prescribed mean l2 error für die Stokes DEIM für die kollaterale Basis (wenn 0, dann keine DEIM in dieser Variable)
- calculate_errors: True oder False, wenn False, dann wird nur die HAPOD-DEIM gemacht, kein reduziertes Problem und damit auch keine Fehler berechnet
- parameter_sampling_type: 
zum Beispiel
```bash
mpiexec -n 32 python3 cellmodel_hapod_deim.py cell_isolation_experiment 1e-2 1e-3 40 40 1e-4 1e-4 1e-4 1e-11 1e-11 1e-11 True log_and_log_inverted method_of_snapshots False False False
```



dune-gdt-super is a git supermodule which serves as a demonstration and testing
module for [dune-gdt](https://github.com/dune-community/dune-gdt). This module
provides the correct versions of all relevant [DUNE](http://www.dune-project.org)
modules and external libraries as git submodules.

In order to build everything, do the following:

* Initialize all submodules:

  ```
  git submodule update --init --recursive
  ```

* Take a look at `config.opts/` and find settings and a compiler which suits your
  system, e.g. `config.opts/gcc-debug`. Select one of those options by defining

  ```
  export OPTS=gcc-debug
  ```
  If you have the `ninja` generator installed we recommend to make use of it by
  selecting `OPTS=gcc-debug.ninja`, which usually speeds up builds significantly.

* Call

  ```
  ./local/bin/gen_path.py
  ```

  to generate a file `PATH.sh` which defines a local build environment. From now
  on you should source this file whenever you plan to work on this project, e.g.:

  ```
  source PATH.sh
  ```

* Download and build all external libraries by calling (this _might_ take some time):

  ```
  ./local/bin/download_external_libraries.py
  ./local/bin/build_external_libraries.py
  ```

* Build all DUNE modules using `cmake` and the selected options (this _will_ take
  some time):

  ```
  ./dune-common/bin/dunecontrol --opts=config.opts/$OPTS --builddir=$PWD/build-$OPTS all
  ```

  This creates a directory corresponding to the selected options (e.g. `build-gcc-debug`)
  which contains a subfolder for each DUNE module. See the `dune/gdt/test` subfolder for
  tests, e.g.,

  ```
  build-gcc-debug/dune-gdt/dune/gdt/test/
  ```
