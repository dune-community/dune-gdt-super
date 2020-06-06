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
from hapod.mpi import MPIWrapper, idle_wait


def coordinatetransformedmn_hapod(
    grid_size, l2_tol, testcase, eval_l2_tol=None, omega=0.95, logfile=None, incremental_gramian=True, orthonormalize=True
):

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
    idle_wait(mpi.comm_world)
    elapsed_data_gen = timer() - start
    times = np.array(times)
    num_snapshots = len(snapshots)
    assert len(times) == num_snapshots
    if eval_l2_tol is not None:
        nonlin_snapshots = snapshots[1:] - snapshots[:-1]

    # Setup HAPOD parameters and input
    # Whether to use an binary tree for the PODs on node level and final POD on rank 0. Setting this to True should improve performance and memory usage.
    use_binary_tree_hapod = True
    # use_binary_tree_hapod = False
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
        hapod_params.append(HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=eval_l2_tol, omega=omega))
        snapshots.append(nonlin_snapshots)
        num_snapshots.append(len(nonlin_snapshots))
        del nonlin_snapshots

    # perform POD of the local trajectory
    modes = [None] * 2 if eval_l2_tol is not None else [None]
    svals = [None] * 2 if eval_l2_tol is not None else [None]
    max_num_local_modes = [None] * 2 if eval_l2_tol is not None else [None]
    max_num_input_vecs = [None] * 2 if eval_l2_tol is not None else [None]
    for i, snaps in enumerate(snapshots):
        max_num_input_vecs[i] = len(snaps)
        modes[i], svals[i] = local_pod(
            [snaps], num_snapshots[i], hapod_params[i], incremental_gramian=False, orthonormalize=orthonormalize, root_of_tree=(rooted_tree_depth == 1)
        )
        max_num_local_modes[i] = len(modes[i])
    del snaps
    del snapshots

    # Now perform binary tree hapods (or simple PODs with the gathered modes if use_binary_tree_hapod == False)
    # over processor cores and compute nodes
    num_snapshots_in_leafs = num_snapshots.copy()
    # For each of the (HA)PODs, we need to collect the communicator, whether this is the last POD in the HAPOD tree,
    # and if this MPI rank participates in the POD. This information is stored in binary_tree_hapods.
    # The empty list added is used to store timings.
    binary_tree_hapods = []
    if mpi.size_proc > 1:
        binary_tree_hapods.append((mpi.comm_proc, mpi.size_rank_0_group == 1, True, []))
    if mpi.size_rank_0_group > 1:
        binary_tree_hapods.append((mpi.comm_rank_0_group, True, mpi.comm_proc == 0, []))

    num_snapshots_in_leafs[i] = num_snapshots[i]
    for comm, root_of_tree_cond, rank_cond, timings in binary_tree_hapods:
        if rank_cond:
            for i in range(len(modes)):
                timings.append(timer())
                if use_binary_tree_hapod:
                    modes[i], svals[i], num_snapshots_in_leafs[i], num_input_vecs, num_local_modes = binary_tree_hapod_over_ranks(
                        comm,
                        modes[i],
                        num_snapshots_in_leafs[i],
                        hapod_params[i],
                        svals=svals[i],
                        last_hapod=root_of_tree_cond,
                        incremental_gramian=incremental_gramian,
                        orthonormalize=orthonormalize,
                    )
                    max_num_input_vecs[i] = max(max_num_input_vecs[i], num_input_vecs)
                    max_num_local_modes[i] = max(max_num_local_modes[i], num_local_modes)
                else:
                    gathered_modes, gathered_svals, num_snapshots_in_leafs[i], _ = comm.gather_on_rank_0(
                        modes[i], num_snapshots[i], svals=svals[i], num_modes_equal=False, merge=False
                    )
                    if comm.rank == 0:
                        pod_inputs = [[m, s] for m, s in zip(gathered_modes, gathered_svals)]
                        max_num_input_vecs[i] = max(max_num_input_vecs[i], sum([len(m) for m in gathered_modes]))
                        modes[i], svals[i] = local_pod(
                            pod_inputs,
                            num_snapshots_in_leafs[i],
                            hapod_params[i],
                            orthonormalize=orthonormalize,
                            incremental_gramian=incremental_gramian,
                            root_of_tree=root_of_tree_cond,
                        )
                        max_num_local_modes = max(max_num_local_modes[i], len(modes[i]))
                        del pod_inputs
                    del gathered_modes
                idle_wait(comm)
                timings[i] = timer() - timings[i]
    idle_wait(mpi.comm_world)

    # write statistics to file
    max_num_local_modes = mpi.comm_world.gather(max_num_local_modes, root=0)
    max_num_input_vecs = mpi.comm_world.gather(max_num_input_vecs, root=0)
    if logfile is not None and mpi.rank_world == 0:
        max_num_local_modes_snaps = max([m[0] for m in max_num_local_modes])
        max_num_input_vecs_snaps = max([m[0] for m in max_num_input_vecs])
        logfile.write(f"The HAPOD resulted in {len(modes[0])} final modes taken from a total of {num_snapshots_in_leafs[0]} snapshots!\n")
        logfile.write(f"max_num_input_vecs = {max_num_input_vecs_snaps}, max_num_local_modes = {max_num_local_modes_snaps}\n")
        if eval_l2_tol is not None:
            max_num_local_modes_eval = max([m[1] for m in max_num_local_modes])
            max_num_input_vecs_eval = max([m[1] for m in max_num_input_vecs])
            logfile.write(f"The DEIM-HAPOD resulted in {len(modes[1])} final modes taken from a total of {num_snapshots_in_leafs[1]} nonlinear snapshots!\n")
            logfile.write(f"max_num_input_vecs = {max_num_input_vecs_eval}, max_num_local_modes = {max_num_local_modes_eval}\n")
        logfile.write(
            "The maximum amount of memory used on rank 0 was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 ** 2) + " GB\n"
        )
        logfile.write("Timings:\n")
        logfile.write("data_gen snaps_proc eval_procs snaps_nodes eval_nodes all\n")
        all_timings = []
        for _, _, _, timings in binary_tree_hapods:
            all_timings += timings
        all_timings = all_timings + [0] * (4 - len(all_timings))
        logfile.write(f"{elapsed_data_gen}" + (" {}" * 5).format(*all_timings, timer()-start) + "\n")

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
    if eval_L2_tol is not None:
        eval_modes, eval_win = mpi.shared_memory_bcast_modes(eval_modes, returnlistvectorarray=True)
    calculate_mean_errors(modes, grid_size, mu, testcase, num_snaps, mpi, logfile=logfile, final_eval_modes=eval_modes, num_evals=eval_num_snaps)
    win.Free()
    if eval_L2_tol is not None:
        eval_win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
