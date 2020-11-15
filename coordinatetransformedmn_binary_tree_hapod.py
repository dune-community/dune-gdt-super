import math
import resource
import sys
from timeit import default_timer as timer
from mpi4py import MPI
from enum import Enum, auto

import numpy as np
from pymor.algorithms.pod import pod

from hapod.coordinatetransformedmn.utility import (
    create_and_scatter_parameters,
    convert_L2_l2,
    calculate_mean_errors,
    create_coordinatetransformedmn_solver,
)
from hapod.coordinatetransformedmn.wrapper import CoordinateTransformedmnOperator
from hapod.hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth
from hapod.mpi import MPIWrapper, idle_wait


class TrajectoryStrategy(Enum):
    """
    Describes possible strategies to handle the adaptive timestepping.

    Possible values:
    USE_FULL_TRAJECTORY: Always compute the full trajectory before computing the first POD in the HAPOD tree. Simple, but may result in a very large first POD. The parameter n is ignored.
    COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS: Computes the trajectory once without storing any results to compute the number of vectors in the trajectory. In the second run, we compute n vectors at a time and then perform a local POD.
    KEEP_LARGEST_DT_IN_INTERVALS: Splits the time interval in subintervals and allow a maximum of n vectors per subinterval. If n is exceeded, only keeps the n vectors with largest associated timestep.
    """

    USE_FULL_TRAJECTORY = auto()
    COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS = auto()
    KEEP_LARGEST_DT_IN_INTERVALS = auto()


def coordinatetransformedmn_hapod(
    grid_size,
    l2_tol,
    testcase,
    eval_l2_tol=None,
    omega=0.95,
    logfile=None,
    incremental_gramian=True,
    orth_tol=1e-10,
    use_binary_tree_hapod=True,
    trajectory_strategy=TrajectoryStrategy.USE_FULL_TRAJECTORY,
    n=None,
):
    """
    Parameters
    ----------
    use_binary_tree_hapod: bool. Whether to use an binary tree for the PODs on node level and final POD on rank 0. Setting this to True should improve performance and memory usage.
    trajectory_strategy: See TrajectoryStrategy.
    """

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mu = create_and_scatter_parameters(testcase, mpi.comm_world, min_param=min_param, max_param=max_param)

    # compute rooted tree depth
    # first, compute contribution of trajectory
    if trajectory_strategy == TrajectoryStrategy.USE_FULL_TRAJECTORY:
        trajectory_tree_depth = 1
    elif trajectory_strategy == TrajectoryStrategy.COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS:
        # solve without saving solution to determine number of timesteps
        solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
        _, snaps, _ = solver.solve(store_operator_evaluations=False, do_not_save=True)
        assert len(snaps) == 0
        del snaps
        num_snapshots = solver.num_timesteps() + 1  # +1 for initial_values
        chunk_size = n
        trajectory_tree_depth = num_snapshots // chunk_size
        if num_snapshots % chunk_size:
            trajectory_tree_depth += 1
    elif trajectory_strategy == TrajectoryStrategy.KEEP_LARGEST_DT_IN_INTERVALS:
        max_vecs_per_interval = n
        # TODO
    else:
        raise NotImplementedError("Unknown TrajectoryStrategy")
    # now compute contribution of higher level PODS
    rooted_tree_depth = trajectory_tree_depth
    if trajectory_strategy == TrajectoryStrategy.USE_FULL_TRAJECTORY and mpi.size_proc > 1:
        # if there is more than one process on a compute node, we perform a second POD on this node
        rooted_tree_depth += 1 if not use_binary_tree_hapod else binary_tree_depth(mpi.comm_proc)
    elif trajectory_strategy == TrajectoryStrategy.COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS:
        # +1 for the local PODs with each chunk
        rooted_tree_depth += 1
    # if there are several compute nodes, we want to compute a final HAPOD
    if mpi.size_world > mpi.size_proc:
        rooted_tree_depth += 1 if not use_binary_tree_hapod else binary_tree_depth(mpi.comm_rank_0_group)
    rooted_tree_depth = mpi.comm_world.allreduce(rooted_tree_depth, op=MPI.MAX)
    # store hapod parameters for easier handling
    hapod_params = [HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=l2_tol, omega=omega)]
    if eval_l2_tol is not None:
        hapod_params.append(HapodParameters(rooted_tree_depth=rooted_tree_depth, epsilon_ast=eval_l2_tol, omega=omega))

    modes = [None] * 2 if eval_l2_tol is not None else [None]
    svals = [None] * 2 if eval_l2_tol is not None else [None]
    max_num_local_modes = [0] * 2 if eval_l2_tol is not None else [0]
    max_num_input_vecs = [0] * 2 if eval_l2_tol is not None else [0]

    # now calculate trajectory
    start = timer()
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
    if trajectory_strategy == TrajectoryStrategy.USE_FULL_TRAJECTORY:
        times, snapshots, nonlinear_snapshots = solver.solve(store_operator_evaluations=True)
        idle_wait(mpi.comm_world)
        times = np.array(times)
        assert len(times) == len(snapshots)
        num_snapshots_in_leafs = [len(snapshots)]
        snapshots = [snapshots]
        if eval_l2_tol is not None:
            # TODO: remove the scaling? gives large errors due to the very large entries of nonlinear_snapshots_alpha
            nonlinear_snapshots.scal(1.0 / nonlinear_snapshots.l2_norm())
            snapshots.append(nonlinear_snapshots)
            num_snapshots_in_leafs.append(len(nonlinear_snapshots))
            del nonlinear_snapshots

        # perform POD of the local trajectory
        for i, snaps in enumerate(snapshots):
            max_num_input_vecs[i] = len(snaps)
            modes[i], svals[i] = local_pod(
                [snaps],
                num_snapshots_in_leafs[i],
                hapod_params[i],
                incremental_gramian=False,
                orth_tol=orth_tol,
                root_of_tree=(rooted_tree_depth == 1),
            )
            max_num_local_modes[i] = len(modes[i])
        del snaps
        del snapshots
    elif trajectory_strategy == TrajectoryStrategy.COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS:
        dt = solver.initial_dt()
        num_snapshots_in_leafs = [0, 0] if eval_l2_tol is not None else [0]
        next_modes = [None] * 2 if eval_l2_tol is not None else [None]
        next_svals = [None] * 2 if eval_l2_tol is not None else [None]
        num_snapshots_in_chunk = [None] * 2 if eval_l2_tol is not None else [None]
        trajectory_tree_depth = mpi.comm_world.allreduce(trajectory_tree_depth, op=MPI.MAX)
        for _ in range(trajectory_tree_depth):
            _, next_snapshots, next_nonlinear_snapshots, dt = solver.next_n_steps(chunk_size, dt, store_operator_evaluations=True)
            assert len(next_snapshots) == chunk_size or solver.finished()
            num_snapshots_in_chunk[0] = len(next_snapshots)
            next_snapshots = [next_snapshots]
            if eval_l2_tol is not None:
                next_snapshots.append(next_nonlinear_snapshots)
                num_snapshots_in_chunk[1] = len(next_nonlinear_snapshots)

            solvers_finished = mpi.comm_proc.reduce(solver.finished(), op=MPI.MIN, root=0)
            for i, snaps in enumerate(next_snapshots):
                max_num_input_vecs[i] = max(max_num_input_vecs[i], len(snaps))
                # calculate POD of timestep vectors on each core
                next_modes[i], next_svals[i] = local_pod(
                    [next_snapshots[i]],
                    num_snapshots_in_chunk[i],
                    hapod_params[i],
                    incremental_gramian=False,
                    orth_tol=orth_tol,
                    root_of_tree=False,
                )
                max_num_local_modes[i] = max(max_num_local_modes[i], len(next_modes[i]))
                del snaps

                gathered_modes, gathered_svals, total_num_snapshots_in_chunk, _ = mpi.comm_proc.gather_on_rank_0(
                    next_modes[i], num_snapshots_in_chunk[i], svals=next_svals[i], num_modes_equal=False, merge=False
                )
                if mpi.rank_proc == 0:
                    num_snapshots_in_leafs[i] += total_num_snapshots_in_chunk
                    pod_inputs = [[m, s] for m, s in zip(gathered_modes, gathered_svals)]
                    if modes[i] is not None:
                        pod_inputs.append([modes[i], svals[i]])
                    max_num_input_vecs[i] = max(max_num_input_vecs[i], sum([len(m) for m, s in pod_inputs]))
                    modes[i], svals[i] = local_pod(
                        pod_inputs,
                        num_snapshots_in_leafs[i],
                        hapod_params[i],
                        orth_tol=orth_tol,
                        incremental_gramian=incremental_gramian,
                        root_of_tree=(mpi.size_proc == mpi.size_world and solvers_finished),
                    )
                    max_num_local_modes[i] = max(max_num_local_modes[i], len(modes[i]))
                    del pod_inputs
                del gathered_modes
        del next_modes
    elif trajectory_strategy == TrajectoryStrategy.KEEP_LARGEST_DT_IN_INTERVALS:
        pass
        # TODO
    else:
        raise NotImplementedError("Unknown TrajectoryStrategy")
    elapsed_data_gen = timer() - start

    # Now perform binary tree hapods (or simple PODs with the gathered modes if use_binary_tree_hapod == False)
    # over processor cores and compute nodes
    # For each of the (HA)PODs, we need to collect the communicator, whether this is the last POD in the HAPOD tree,
    # and if this MPI rank participates in the POD. This information is stored in binary_tree_hapods.
    # The empty list added is used to store timings.
    binary_tree_hapods = []
    if trajectory_strategy == TrajectoryStrategy.USE_FULL_TRAJECTORY and mpi.size_proc > 1:
        binary_tree_hapods.append((mpi.comm_proc, mpi.size_rank_0_group == 1, True, []))
    if mpi.size_rank_0_group > 1:
        binary_tree_hapods.append((mpi.comm_rank_0_group, True, mpi.comm_proc == 0, []))
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
                        orth_tol=orth_tol,
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
                            orth_tol=orth_tol,
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
            logfile.write(
                f"The DEIM-HAPOD resulted in {len(modes[1])} final modes taken from a total of {num_snapshots_in_leafs[1]} nonlinear snapshots!\n"
            )
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
        logfile.write(f"{elapsed_data_gen}" + (" {}" * 5).format(*all_timings, timer() - start) + "\n")

    if eval_l2_tol is not None:
        return modes[0], svals[0], num_snapshots_in_leafs[0], modes[1], svals[1], num_snapshots_in_leafs[1], mu, mpi
    else:
        return modes[0], svals[0], num_snapshots_in_leafs[0], None, None, None, mu, mpi


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-1 if argc < 3 else float(sys.argv[2])
    eval_L2_tol = None if argc < 4 else (None if sys.argv[3] == "None" else float(sys.argv[3]))
    testcase = "HFM50SourceBeam" if argc < 5 else sys.argv[4]
    omega = 0.95 if argc < 6 else float(sys.argv[5])
    inc_gramian = True if argc < 7 else not (sys.argv[6] == "False" or sys.argv[6] == "0")
    filename = f"{testcase}_HAPOD_gridsize_{grid_size}_tol_{L2_tol}_evaltol_{eval_L2_tol}_omega_{omega}.log"
    logfile = open(filename, "a")
    modes, svals, num_snaps, eval_modes, eval_svals, eval_num_snaps, mu, mpi = coordinatetransformedmn_hapod(
        grid_size=grid_size,
        l2_tol=convert_L2_l2(L2_tol, grid_size, testcase),
        testcase=testcase,
        eval_l2_tol=convert_L2_l2(eval_L2_tol, grid_size, testcase) if eval_L2_tol is not None else None,
        omega=omega,
        logfile=logfile,
        incremental_gramian=inc_gramian,
        trajectory_strategy=TrajectoryStrategy.COMPUTE_TWICE_TO_SPLIT_IN_CHUNKS,
        n=10,
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
