import resource
import sys
from timeit import default_timer as timer

import numpy as np
from pymor.algorithms.pod import pod

from boltzmannutility import calculate_error, create_and_scatter_cellmodel_parameters, create_cellmodel_solver
from mpiwrapper import MPIWrapper
from boltzmann.wrapper import CellModelPfieldProductOperator, CellModelOfieldProductOperator, CellModelStokesProductOperator

def calculate_pod(result, product):
    num_snapshots = len(result)

    # gather snapshots on rank 0
    result, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(result, num_snapshots, num_modes_equal=True)
    svals = None

    # perform a POD
    elapsed_pod = 0
    if mpi.rank_world == 0:
        result, svals = pod(result, product=product, atol=0., rtol=0., l2_err=tol * np.sqrt(total_num_snapshots))
        elapsed_pod = timer() - start
    return result, svals, elapsed_pod, total_num_snapshots

def cellmodel_pod(grid_size_y, tol, logfile=None):
    t_end = 0.03
    dt = 0.001
    write_step = dt

    # get MPI communicators
    mpi = MPIWrapper()

    # get cellmodel solver to create snapshots
    mu = create_and_scatter_cellmodel_parameters(mpi.comm_world)
    solver = create_cellmodel_solver('single_cell', grid_size_y * 4, grid_size_y, mu)

    # calculate Boltzmann problem trajectory
    start = timer()
    snapshots_pfield, snapshots_ofield, snapshots_stokes = solver.solve(t_end, dt, write_step, 'dings', True)
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start

    # perform pod
    CellModelPfieldProductOperator pfield_product(solver);
    CellModelOfieldProductOperator ofield_product(solver);
    CellModelStokesProductOperator stokes_product(solver);

    result_pfield, svals_pfield, elapsed_pod_pfield, total_num_snapshots_pfield = calculate_pod(snapshots_pfield, pfield_product);
    result_ofield, svals_ofield, elapsed_pod_ofield, total_num_snapshots_ofield = calculate_pod(snapshots_ofield, ofield_product);
    result_stokes, svals_stokes, elapsed_pod_stokes, total_num_snapshots_stokes = calculate_pod(snapshots_stokes, stokes_product);

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("After the pfield POD, there are " + str(len(result_pfield)) + " modes of " +
                      str(total_num_snapshots_pfield) + " snapshots left!\n")
        logfile.write("After the ofield POD, there are " + str(len(result_ofield)) + " modes of " +
                      str(total_num_snapshots_ofield) + " snapshots left!\n")
        logfile.write("After the stokes POD, there are " + str(len(result_stokes)) + " modes of " +
                      str(total_num_snapshots_stokes) + " snapshots left!\n")
        logfile.write("The maximum amount of memory used on rank 0 was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB\n")
        elapsed = timer() - start
        logfile.write("Time elapsed: " + str(elapsed) + "\n")

    return result_pfield, svals_pfield, total_num_snapshots_pfield, result_ofield, svals_ofield, total_num_snapshots_pfield, result_stokes, svals_stokes, total_num_snapshots_stokes, mu, mpi


if __name__ == "__main__":
    grid_size_y = int(sys.argv[1])
    tol = float(sys.argv[2])
    filename = "cellmodel_POD_gridsize_%d_tol_%f" % (grid_size_y, tol)
    logfile = open(filename, "a")
    final_modes_pfield, _, total_num_snapshots_pfield, final_modes_ofield, _, total_num_snapshots_ofield, final_modes_stokes, _, total_num_snapshots_stokes, mu, mpi = cellmodel_pod(grid_size_y, tol * grid_size_y, logfile=logfile)
    final_modes_pfield, win_pfield = mpi.shared_memory_bcast_modes(final_modes_pfield)
    calculate_error(final_modes_pfield, total_num_snapshots_pfield, final_modes_ofield, total_num_snapshots_ofield,
                    final_modes_stokes, total_num_snapshots_stokes, grid_size_y, mu, mpi, logfile=logfile)
    win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
