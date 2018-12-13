import resource
import sys
from timeit import default_timer as timer

import numpy as np
from pymor.algorithms.pod import pod

from boltzmannutility import calculate_error, create_and_scatter_boltzmann_parameters, create_boltzmann_solver
from mpiwrapper import MPIWrapper


def boltzmann_pod(grid_size, tol, logfile=None):

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    mu = create_and_scatter_boltzmann_parameters(mpi.comm_world)
    solver = create_boltzmann_solver(grid_size, mu)

    # calculate Boltzmann problem trajectory
    start = timer()
    result = solver.solve()
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start
    num_snapshots = len(result)

    # gather snapshots on rank 0
    result, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(result, num_snapshots, num_modes_equal=True)
    svals = None

    # perform a POD
    elapsed_pod = 0
    if mpi.rank_world == 0:
        result, svals = pod(result, atol=0., rtol=0., l2_err=tol*np.sqrt(total_num_snapshots))
        elapsed_pod = timer() - start

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
            logfile.write("After the POD, there are " + str(len(result)) + " modes of "
                          + str(total_num_snapshots) + " snapshots left!\n")
            logfile.write("The maximum amount of memory used on rank 0 was: " +
                          str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
            elapsed = timer() - start
            logfile.write("Time elapsed: " + str(elapsed) + "\n")

    return result, svals, total_num_snapshots, mu, mpi, elapsed_data_gen, elapsed_pod, solver


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    tol = float(sys.argv[2])
    filename = "POD_gridsize_%d_tol_%f" % (grid_size, tol)
    logfile = open(filename, "a")
    final_modes, _, total_num_snapshots, mu, mpi, _, _, _ = boltzmann_pod(grid_size, tol * grid_size, logfile=logfile)
    final_modes, win = mpi.shared_memory_bcast_modes(final_modes)
    calculate_error(final_modes, grid_size, mu, total_num_snapshots, mpi, logfile=logfile)
    win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
