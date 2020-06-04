import math
import resource
import sys
from timeit import default_timer as timer
from mpi4py import MPI

import numpy as np
from pymor.algorithms.pod import pod

from hapod.coordinatetransformedmn.utility import (
    create_and_scatter_parameters,
    convert_L2_l2,
    calculate_mean_errors,
    create_coordinatetransformedmn_solver,
)
from hapod.hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth
from hapod.mpi import MPIWrapper


def coordinatetransformedmn_hapod(
    grid_size, l2_tol, testcase, eval_l2_tol=None, omega=0.95, logfile=None, incremental_gramian=True, orthonormalize=True
):

    start = timer()

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mu = create_and_scatter_parameters(testcase, mpi.comm_world, min_param=min_param, max_param=max_param)
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)

    # calculate problem trajectory
    start = timer()
    times, snapshots = solver.solve()
    times = np.array(times)
    elapsed_data_gen = timer() - start
    num_snapshots = len(snapshots)
    assert len(times) == num_snapshots
    if eval_l2_tol is not None:
        nonlin_snapshots = snapshots[1:] - snapshots[:-1]

    # Setup HAPOD parameters and input
    # Whether to use an binary tree for the PODs on node level and final POD on rank 0. Setting this to True should improve performance and memory usage.
    use_binary_tree_hapod = True
    # Calculate depth of rooted tree
    # Start with 1 for calculation of the trajectories
    rooted_tree_depth = 1
    # if there is more than one process on a compute node, we perform a second POD on this node
    if mpi.size_proc > 1:
        rooted_tree_depth += 1 if not use_binary_tree_hapod else binary_tree_depth(mpi.comm_proc)
    # if there are several compute nodes, we want to compute a final HAPOD
    if mpi.size_world > mpi.size_proc:
        rooted_tree_depth += 1 if not use_binary_tree_hapod else binary_tree_depth(mpi.comm_rank_0_group)
    rooted_tree_depth = mpi.comm_world.allreduce(rooted_tree_depth, op=MPI.MAX)
    snapshots = [snapshots]
    num_snapshots = [num_snapshots]
    hapod_params = [HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=l2_tol, omega=omega)]
    if eval_l2_tol is not None:
        hapod_params.append(HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=eval_l2_tol, omega=omega));
        snapshots.append(nonlin_snapshots)
        num_snapshots.append(len(nonlin_snapshots))
        del nonlin_snapshots

    # perform POD of the local trajectory
    modes = [None] * 2 if eval_l2_tol is not None else [None]
    svals = [None] * 2 if eval_l2_tol is not None else [None]
    for i, snaps in enumerate(snapshots):
        modes[i], svals[i] = local_pod(
            [snaps], num_snapshots[i], hapod_params[i], incremental_gramian=False, orthonormalize=orthonormalize, root_of_tree=(rooted_tree_depth == 1)
        )
    del snaps
    del snapshots

    # gather modes on each compute node and perform second level PODs
    start_proc = timer()
    num_snapshots_in_leafs = num_snapshots.copy()
    binary_trees = []
    if mpi.size_proc > 1:
        binary_trees.append((mpi.comm_proc, mpi.size_rank_0_group == 1, True))
    if mpi.size_rank_0_group > 1:
        binary_trees.append((mpi.comm_rank_0_group, True, mpi.comm_proc == 0))

    num_snapshots_in_leafs[i] = num_snapshots[i]
    for comm, root_of_tree_cond, rank_cond in binary_trees:
        if rank_cond:
            for i in range(len(modes)):
                if use_binary_tree_hapod:
                    modes[i], svals[i], num_snapshots_in_leafs[i], _, _ = binary_tree_hapod_over_ranks(
                        comm,
                        modes[i],
                        num_snapshots_in_leafs[i],
                        hapod_params[i],
                        svals=svals[i],
                        last_hapod=root_of_tree_cond,
                        incremental_gramian=incremental_gramian,
                        orthonormalize=orthonormalize,
                    )
                else:
                    gathered_modes, gathered_svals, num_snapshots_in_leafs[i], _ = comm.gather_on_rank_0(
                        modes[i], num_snapshots[i], svals=svals[i], num_modes_equal=False, merge=False
                    )
                    if comm.rank == 0:
                        pod_inputs = [[m, s] for m, s in zip(gathered_modes, gathered_svals)]
                        modes[i], svals[i] = local_pod(
                            pod_inputs,
                            num_snapshots_in_leafs[i],
                            hapod_params[i],
                            orthonormalize=orthonormalize,
                            incremental_gramian=incremental_gramian,
                            root_of_tree=root_of_tree_cond,
                        )
                        del pod_inputs
                    del gathered_modes

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write(f"The HAPOD resulted in {len(modes[0])} final modes taken from a total of {num_snapshots_in_leafs[0]} snapshots!\n")
        if eval_l2_tol is not None:
            logfile.write(f"The DEIM-HAPOD resulted in {len(modes[1])} final modes taken from a total of {num_snapshots_in_leafs[1]} nonlinear snapshots!\n")
        logfile.write(
            "The maximum amount of memory used on rank 0 was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 ** 2) + " GB\n"
        )
        # logfile.write("Max time for HAPOD on processors:" + str(elapsed_proc) + "\n")
        # logfile.write("Time for final HAPOD over nodes:" + str(elapsed_final) + "\n")
        logfile.write("Time for all:" + str(timer() - start) + "\n")

    return modes[0], svals[0], num_snapshots_in_leafs[0], *([modes[1], svals[1], num_snapshots_in_leafs[1]] if eval_l2_tol is not None else [None] * 3), mu, mpi


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-1 if argc < 3 else float(sys.argv[2])
    eval_L2_tol = None if argc < 4 else float(sys.argv[3])
    testcase = "HFM50SourceBeam" if argc < 5 else sys.argv[4]
    omega = 0.95 if argc < 6 else float(sys.argv[5])
    inc_gramian = True if argc < 7 else not (sys.argv[6] == "False" or sys.argv[6] == "0")
    filename = f"{testcase}_HAPOD_gridsize_{grid_size}_tol_{L2_tol}_omega_{omega}.log"
    logfile = open(filename, "a")
    modes, svals, num_snaps, eval_modes, eval_svals, eval_num_snaps, mu, mpi = coordinatetransformedmn_hapod(
        grid_size=grid_size,
        l2_tol=convert_L2_l2(L2_tol, grid_size, testcase),
        testcase=testcase,
        eval_l2_tol=convert_L2_l2(eval_L2_tol, grid_size, testcase) if eval_L2_tol is not None else None,
        omega=omega,
        logfile=logfile,
        incremental_gramian=inc_gramian,
    )
    modes, win = mpi.shared_memory_bcast_modes(modes, returnlistvectorarray=True)
    eval_modes, eval_win = mpi.shared_memory_bcast_modes(eval_modes, returnlistvectorarray=True)
    calculate_mean_errors(modes, grid_size, mu, testcase, num_snaps, mpi, logfile=logfile, final_eval_modes=eval_modes, num_evals=eval_num_snaps)
    win.Free()
    eval_win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
