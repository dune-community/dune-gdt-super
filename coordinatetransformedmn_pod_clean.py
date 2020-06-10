import sys
import numpy as np
from pymor.algorithms.pod import pod

from hapod.coordinatetransformedmn.utility import (
    create_and_scatter_parameters,
    convert_L2_l2,
    calculate_mean_errors,
    create_coordinatetransformedmn_solver,
)
from hapod.mpi import MPIWrapper, idle_wait


def coordinatetransformedmn_pod(grid_size, l2_tol, testcase, logfile=None):

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mu = create_and_scatter_parameters(testcase, mpi.comm_world, min_param=min_param, max_param=max_param)
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)

    # calculate problem trajectory
    times, snapshots, nonlinear_snapshots = solver.solve(store_operator_evaluations=True)
    num_snapshots = len(snapshots)
    assert len(times) == num_snapshots
    del solver

    # gather snapshots on rank 0
    snapshots, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(snapshots, num_snapshots, num_modes_equal=False)

    # perform a POD
    svals = None
    if mpi.rank_world == 0:
        snapshots, svals = pod(snapshots, atol=0.0, rtol=0.0, l2_err=l2_tol * np.sqrt(total_num_snapshots))
        if logfile is not None:
            logfile.write("After the POD, there are " + str(len(snapshots)) + " modes of " + str(total_num_snapshots) + " snapshots left!\n")
    idle_wait(mpi.comm_world)

    return snapshots, svals, total_num_snapshots, mu, mpi


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-1 if argc < 3 else float(sys.argv[2])
    testcase = "HFM50SourceBeam" if argc < 4 else sys.argv[3]
    filename = f"{testcase}_POD_gridsize_{grid_size}_tol_{L2_tol}.log"
    logfile = open(filename, "a")
    final_modes, _, total_num_snapshots, mu, mpi = coordinatetransformedmn_pod(
        grid_size, convert_L2_l2(L2_tol, grid_size, testcase), testcase, logfile=logfile
    )
    final_modes, win = mpi.shared_memory_bcast_modes(final_modes, returnlistvectorarray=True)
    calculate_mean_errors(final_modes, grid_size, mu, testcase, total_num_snapshots, mpi, logfile=logfile)
    win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
