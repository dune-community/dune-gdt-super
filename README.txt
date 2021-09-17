This archive contains the code used to obtain the results presented in "A new entropy-variable-based discretization method for minimum
entropy moment approximations of linear kinetic equations" by Tobias Leibner and Mario Ohlberger.
The code is based on dune-gdt (DUNE generic discretization toolbox, https://github.com/dune-community/dune-gdt), a discretization module for
the C++ software framework DUNE (Distributed and Unified Numerics Environment, https://www.dune-project.org/). All required DUNE modules
in the correct versions are provided with these supplementary files in the dune-gdt-super folder.
The code is probably not compatible to other (including newer) versions of the DUNE modules.

To run the code yourself, you need to (tested on Ubuntu 20.04)
1. Install the required dependencies: build-essential, gfortran, git, CMake, python3, pip, pkg-config. This can be done e.g. via

apt-get install build-essential gfortran git cmake python3-dev python3-pip pkg-config

2. Change to the dune-gdt-super folder, download and install some more required external packages by running

cd dune-gdt-super
CC=gcc ./bin/download_external_libraries.py
CC=gcc ./bin/build_external_libraries.py

This downloads and builds the Boost (v1.77.0), Eigen (v3.4.0), qhull (v8.0.2) and Coin-or Clp (v1.17.6) libraries.
For the new entropy-variable-based scheme, only Boost and Eigen are required, qhull and Clp are only used for the reference schemes.
If the provided links are not valid anymore, you can install these libraries in the required versions yourself.
In that case, you have to replace the first line in step 3 below by
export BOOST_ROOT=/path/to/boost/root
where path/to/boost/root/ is the directory containing the lib and include folders for Boost.

3. Build all DUNE modules by running

export BOOST_ROOT=$(pwd)/environments/debian-minimal/local/
./dune-common/bin/dunecontrol --opts=config.opts/gcc-release all

4. Edit the file dune-gdt/dune/gdt/test/momentmodels/hyperbolic__momentmodels__entropic_coords_mn.cc to only
include the test cases you are about (delete or comment the other lines, make sure to have a comma after each line, except
for the last line). Exemplary lines for one-dimensional test cases are

Dune::GDT::PlaneSourceMnTestCase<Yasp1, Dune::GDT::LegendreMomentBasis<double, double, 7>, false, true>,
Dune::GDT::PlaneSourceMnTestCase<Yasp1, Dune::GDT::HatFunctionMomentBasis<double, 1, double, 8, 1, 1>, false, true>,
Dune::GDT::SourceBeamMnTestCase<Yasp1, Dune::GDT::PartialMomentBasis<double, 1, double, 8, 1, 1>, false, true>,

Here, the number after the second "double" in the basis' template argument gives the moment order N (for M_N) models
or the number of moments n (for PMM_n and HFM_n) models. So if you want to run the source-beam test for the M_10 or the
HFM_10 model, you have to include the lines

Dune::GDT::SourceBeamMnTestCase<Yasp1, Dune::GDT::LegendreMomentBasis<double, double, 10>, false, true>,
Dune::GDT::SourceBeamMnTestCase<Yasp1, Dune::GDT::HatFunctionMomentBasis<double, 1, double, 10, 1, 1>, false, true>,

respectively. For the three-dimensional test cases, that number still specifies the order N for M_N models.
However, for the HFM_n and PMM_n models, that number now specifies the number of refinements of the initial
octahedron used in the construction of the basis, i.e.

Dune::GDT::PointSourceMnTestCase<Yasp3, Dune::GDT::HatFunctionMomentBasis<double, 3, double, 0, 1, 3>, false, true>,
Dune::GDT::PointSourceMnTestCase<Yasp3, Dune::GDT::HatFunctionMomentBasis<double, 3, double, 1, 1, 3>, false, true>,
Dune::GDT::PointSourceMnTestCase<Yasp3, Dune::GDT::HatFunctionMomentBasis<double, 3, double, 2, 1, 3>, false, true>,

denotes the point-source test with HFM_6, HFM_18 and HFM_66 basis, respectively.

If you also want to run the test cases with the reference scheme, you have to make similar changes in the file
dune-gdt/dune/gdt/test/momentmodels/hyperbolic__momentmodels__mn_ord1.cc.

5. Change to the build directory of dune-gdt and compile the relevant executables

cd build/gcc-release/dune-gdt
make test_hyperbolic__momentmodels__entropic_coords_mn
make test_hyperbolic__momentmodels__mn_ord1

6. Run the generated executables, i.e.

./dune/gdt/test/test_hyperbolic__momentmodels__entropic_coords_mn

for the new scheme and

./dune/gdt/test/test_hyperbolic__momentmodels__mn_ord1

for the reference scheme. You can ignore the "[  FAILED  ] ... " output.

The following command line options will be accepted by both executables:
-grid_size INT                      Number of grid cells, INT is an integer, e.g. -grid_size 100 will give a grid with 100 cells in 1d and a grid with 100x100x100 cells in 3d.
                                    For 3d tests, you can alternatively specify the grid size as -grid_size 100x200x300 to get a grid with 100, 200, 300 elements
                                    in x, y and z direction, respectively.
-t_end FLOAT                        Final time t_f, FLOAT is a floating point number, valid formats include -t_end 1, -t_end 0.1, -t_end 1e-1
-threading.max_count INT            Number of computational threads (only available if Intel TBB has been found during the cmake configuration)
-num_save_steps INT                 Number of time points at which output files will be written. INT may be an arbitrary positive integer or -1.
                                    For examples, with -num_save_steps 10 -t_end 1, output files will be written at times t=0, 0.1, 0.2, 0.3, ..., 0.9, 1.0.
                                    If INT = -1, output files will be written for all time steps. Note that using the -num_save_steps INT option with INT different
                                    from -1 may alter the time steps, since time steps will be chosen to exactly reach the specified output time points.
-num_output_steps INT               Similar to -num_save_steps, except that no files will be written but only a message will be printed to the terminal at the
                                    specified time points (useful to see progress for long-running tests). Also, different to the -num_save_steps option,
                                    time steps will not be altered to match the specified times exactly.
-filename STRING                    Prefix for the output files

The following command line options are only available for the new scheme (test_hyperbolic__momentmodels__entropic_coords_mn)
-timestepper.atol FLOAT             Specifies the absolute tolerance for the error computation of the adaptive time stepping scheme (see eq. ??? in the paper)
-timestepper.rtol FLOAT             Specifies the relative tolerance for the error computation of the adaptive time stepping scheme (see eq. ??? in the paper)
-apply_gamma_relaxation BOOL        If BOOL = 1, the relaxed Runge-Kutta scheme will be used (see ??? in the paper)
-massmatrix_regularization FLOAT    Enables regularization by the mass matrix with regularization parameter FLOAT (see ??? in the paper)

The following command line options are only available for the reference scheme (test_hyperbolic__momentmodels__mn_ord1)
-timestepper.dt FLOAT               Specifies the time step to use

For example, the call

./dune/gdt/test/test_hyperbolic__momentmodels__entropic_coords_mn -t_end 1 -num_output_steps -1 -num_save_steps 2 -grid_size 1200 -threading.max_count 1 -filename new_scheme -timestepper.atol 1e-3 -timestepper.rtol 1e-3

will run the new scheme with a tolerance tau=1e-3 for the test cases specified in step 4 with a grid size of 1200 (1d tests) or 1200^3 (3d tests) until the final time 1 and write output files
(new_scheme_*.vtp/vtu for visualization of the local particle density rho and new_scheme_*.txt for the current solution values) at times 0, 0.5 and 1. The files will be numbered consecutively, i.e. the new_scheme_*_0.vtp,
new_scheme_*_1.vtp and new_scheme_*_2.vtp files include the visualization for times 0, 0.5 and 1, respectively. The .txt files contain two columns, the first one denotes the center of the a grid cell and
the second column gives the mean value of the local particle density rho in that grid cell.

The corresponding call for the reference scheme is

./dune/gdt/test/test_hyperbolic__momentmodels__mn_ord1 -t_end 1 -num_output_steps -1 -num_save_steps 2 -grid_size 1200 -threading.max_count 1 -filename reference_scheme

