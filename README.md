```
# This file is part of the dune-gdt-super project:
#   https://github.com/dune-community/dune-gdt-super
# The copyright lies with the authors of this file (see below).
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository contains the code for the continuum model which was used to compute the results presented in
"Dynamic interplay of protrusive microtubule and contractile actomyosin forces drives tissue extension" by
Amrita Singh, Sameedha Thale, Tobias Leibner, Andrea Ricker, Harald Nüsse, Jürgen Klingauf, Mario Ohlberger and Maja Matis.

The code is based on dune-gdt (DUNE generic discretization toolbox, https://github.com/dune-community/dune-gdt), a discretization module for
the C++ software framework DUNE (Distributed and Unified Numerics Environment, https://www.dune-project.org/). All required DUNE modules
in the correct versions are provided in this super module.
The code is probably not compatible to other (including newer) versions of the DUNE modules.

To run the code yourself, you need to (requirements for a fresh version of Ubuntu 20.04)
1. Install the required dependencies: build-essential, gfortran, git, CMake, python3, pip, pkg-config. This can be done e.g. via

```bash
apt-get install build-essential gfortran git cmake python3-dev python3-pip pkg-config
```

2. Change to the dune-gdt-super folder, download and install some more required external packages by running

```bash
cd dune-gdt-super
CC=gcc ./bin/download_external_libraries.py
CC=gcc ./bin/build_external_libraries.py
```

3. Build all DUNE modules by running

```bash
export BOOST_ROOT=$(pwd)/environments/debian-minimal/local/
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release all
```

You can adapt the `config.opts/gcc-release` file to your needs before running these commands, or use, e.g., the `config.opts/clang-release` file
if you want to compile with `clang` instead of `gcc`.

5. Change to the build directory of dune-gdt and compile the relevant executables

```bash
cd build/gcc-release/dune-gdt
make examples__cellmodel
```

6. Link the configuration file to the build directory

```bash
cd ../../..
ln -s $(pwd)/dune-gdt/examples/activepolargels.ini $(pwd)/build/gcc-release/dune-gdt/activepolargels.ini
```

7. Adapt the configuration to the setting you want to simulate by adapting the `activepolargels.ini` file. You might want to adapt:

- grid.NX or grid.NY: Number of grid elements in x and y direction, respectively
- fem.t_end: End time of the simulation
- fem.dt: Time step length
- problem.epsilon: Controls the width of the diffusive interface which models the cell membrane
- problem.Be, problem.Ca, problem.Fa: Value of the parameters Be, Ca, Fa
- output.prefix: Prefix of the generated output files (will have the format prefix\*.vtu)
- output.write_step: Time step for the visualization
- output.subsampling: Whether subsampling should be used for the visualization

8. Run the generated executable, i.e.

```bash
cd build/gcc-release/dune-gdt
./dune/gdt/examples/examples__cellmodel
```





