This archive contains the code used to obtain the results presented in "First-order continuous- and discontinuous-Galerkin moment models for a
linear kinetic equation: model derivation and realizability theory" by Florian Schneider and Tobias Leibner.
The code is based on dune-gdt (DUNE generic discretization toolbox, https://github.com/dune-community/dune-gdt), a discretization module for
the C++ software framework DUNE (Distributed and Unified Numerics Environment, https://www.dune-project.org/). All required DUNE modules
in the correct versions (dune-gdt: 18.10, dune-xt modules: 18.10, DUNE core modules: 2.5) are provided with this supplementary files in
the dune-gdt-super folder. The code is probably not compatible to other (including newer) versions of the DUNE modules.

To run the code yourself, you need to (tested on Ubuntu 18.04 and Bash on Ubuntu 18.04 on Windows 10)
1. Install the required dependencies: build-essential, gfortran, git, CMake, python3, pkg-config. This can be done e.g. via

apt-get install build-essential gfortran git cmake python3-dev pkg-config

2. Change to the dune-gdt-super folder, download and install some more required external packages by running

cd dune-gdt-super
CC=gcc python3 bin/download_external_libraries.py
CC=gcc python3 bin/build_external_libraries.py

This downloads and builds the Boost (v1.70.0) and Eigen (v3.3.7) libraries. If the provided links for Boost
or Eigen are not valid anymore, you can install Boost or Eigen in the required versions yourself. In that case, you have 
to replace the first line in step 3 below by
export BOOST_ROOT=/path/to/boost/root
where path/to/boost/root/ is the directory containing the lib and include folders for Boost.

3. Build all DUNE modules by running

export BOOST_ROOT=$(pwd)/environments/debian-minimal/local/
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release all

4. Change to the build directory of dune-gdt and compile (at least one of) the relevant executables

cd build/gcc-release/dune-gdt
make examples__momentapproximation_1dhatfunctions
make examples__momentapproximation_1dpartialmoments
make examples__momentapproximation_1dfullmoments
make examples__momentapproximation_3dhatfunctions
make examples__momentapproximation_3dpartialmoments
make examples__momentapproximation_3dfullmoments

5. Choose a prescribed distribution (Gauss1d, Heaviside or CrossingBeams1d for 1d basisfunctions, GaussOnSphere, SquareOnSphere or CrossingBeams3d for 3d) and run the executable, e.g.,

./examples/examples__momentapproximation_1dhatfunctions Heaviside
./examples/examples__momentapproximation_1dpartialmoments Gauss1d
./examples/examples__momentapproximation_3dhatfunctions GaussOnSphere
./examples/examples__momentapproximation_3dfullmoments SquareOnSphere

If the code succesfully runs, it will output one .txt file per moment model (e.g. heaviside_1dhfm2.txt, heaviside_1dhfm3.txt, ...
and heaviside_1dhfp2.txt, heaviside_1dhfp3.txt, ... for the first command above). The first column (in 1d) or first three columns (in 3d)
of these text files contain the cartesian coordinates of the quadrature points, the second (1d) or fourth (3d) column contains the
approximated density at that point (see the comment at dune-gdt-super/dune-gdt/dune/gdt/test/hyperbolic/moment-approximation.hh, line 48ff,
for further information on the choice of quadrature points).
You can prescribe a distribution of your choice by adding a branch to the if statement starting in line 151 of moment-approximation.hh
and recompiling.
By default, moment models are calculated only up to a relatively low maximal order. This ensures that the execution should be finished in acceptable
time even on average hardware. To reproduce the high-order results you need to set the variables max_order, max_number_of_moments or max_refinements in
the main function in the .cc file of interest (dune-gdt/examples/momentapproximation_{}.cc) and recompile. For the 3d models, you also need to adjust the
quadrature_refinements.
By default, Maxwell-Boltzmann entropy is used. To use Bose-Einstein entropy, comment the Maxwell-Boltzmann entropy and uncomment the Bose-Einstein entropy
definition in the .cc file.
