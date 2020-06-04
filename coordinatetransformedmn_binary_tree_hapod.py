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

    hapod_params = HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=l2_tol, omega=omega)
    hapod_params = HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=eval_l2_tol, omega=omega)

    # calculate problem trajectory
    start = timer()
    times, snapshots = solver.solve()
    times = np.array(times)
    elapsed_data_gen = timer() - start
    num_snapshots = len(snapshots)
    assert len(times) == num_snapshots

    # perform POD of the local trajectory
    modes, svals = local_pod(
        [snapshots], num_snapshots, hapod_params, incremental_gramian=False, orthonormalize=orthonormalize, root_of_tree=(rooted_tree_depth == 1)
    )
    del snapshots

    # gather modes on each compute node and perform second level PODs
    start_proc = timer()
    if mpi.size_proc > 1:
        if use_binary_tree_hapod:
            modes, svals, num_snapshots_on_compute_node, _, _ = binary_tree_hapod_over_ranks(
                mpi.comm_proc,
                modes,
                num_snapshots,
                hapod_params,
                svals=svals,
                last_hapod=(mpi.size_rank_0_group == 1),
                incremental_gramian=incremental_gramian,
                orthonormalize=orthonormalize,
            )
        else:
            gathered_modes, gathered_svals, num_snapshots_on_compute_node, _ = mpi.comm_proc.gather_on_rank_0(
                modes, num_snapshots, svals=svals, num_modes_equal=False, merge=False
            )
            del modes
            if mpi.rank_proc == 0:
                pod_inputs = [[m, s] for m, s in zip(gathered_modes, gathered_svals)]
                modes, svals = local_pod(
                    # [[gathered_modes, gathered_svals]],
                    pod_inputs,
                    num_snapshots_on_compute_node,
                    hapod_params,
                    orthonormalize=orthonormalize,
                    incremental_gramian=incremental_gramian,
                    root_of_tree=(mpi.size_rank_0_group == 1),
                )
                del gathered_modes
    else:
        num_snapshots_on_compute_node = num_snapshots
    elapsed_proc = timer() - start_proc
    elapsed_proc = mpi.comm_world.reduce(elapsed_proc, op=MPI.MAX, root=0)

    start_final = timer()
    if mpi.size_rank_0_group > 1:
        if mpi.rank_proc == 0:
            if use_binary_tree_hapod:
                # Perform a HAPOD over a binary tree of nodes
                final_modes, svals, total_num_snapshots, _, _ = binary_tree_hapod_over_ranks(
                    mpi.comm_rank_0_group,
                    modes,
                    num_snapshots_on_compute_node,
                    hapod_params,
                    svals=svals,
                    last_hapod=True,
                    incremental_gramian=incremental_gramian,
                    orthonormalize=orthonormalize,
                )
                del modes
            else:
                gathered_modes, gathered_svals, total_num_snapshots, _ = mpi.comm_rank_0_group.gather_on_rank_0(
                    modes, num_snapshots_on_compute_node, svals=svals, num_modes_equal=False
                )
                del modes
                final_modes, svals = local_pod(
                    [gathered_modes, gathered_svals],
                    total_num_snapshots,
                    hapod_params,
                    orthonormalize=orthonormalize,
                    incremental_gramian=incremental_gramian,
                    root_of_tree=True,
                )
                del gathered_modes
    else:
        final_modes, total_num_snapshots = (modes, num_snapshots_on_compute_node) if mpi.rank_proc == 0 else (None, None)
    elapsed_final = timer() - start_final

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write(f"The HAPOD resulted in {len(final_modes)} final modes taken from a total of {total_num_snapshots} snapshots!\n")
        logfile.write(
            "The maximum amount of memory used on rank 0 was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 ** 2) + " GB\n"
        )
        logfile.write("Max time for HAPOD on processors:" + str(elapsed_proc) + "\n")
        logfile.write("Time for final HAPOD over nodes:" + str(elapsed_final) + "\n")
        logfile.write("Time for all:" + str(timer() - start) + "\n")

    return (final_modes, svals, total_num_snapshots, mu, mpi) if mpi.rank_world == 0 else (None, None, None, mu, mpi)


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-3 if argc < 3 else float(sys.argv[2])
    testcase = "HFM50SourceBeam" if argc < 4 else sys.argv[3]
    omega = 0.95 if argc < 5 else float(sys.argv[4])
    inc_gramian = True if argc < 6 else not (sys.argv[5] == "False" or sys.argv[5] == "0")
    filename = f"{testcase}_HAPOD_gridsize_{grid_size}_tol_{L2_tol}_omega_{omega}.log"
    logfile = open(filename, "a")
    final_modes, _, total_num_snapshots, mu, mpi = coordinatetransformedmn_hapod(
        grid_size=grid_size,
        l2_tol=convert_L2_l2(L2_tol, grid_size, testcase),
        testcase=testcase,
        eval_l2_tol=None,
        omega=omega,
        logfile=logfile,
        incremental_gramian=inc_gramian,
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
