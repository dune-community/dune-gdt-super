import resource
import sys
from timeit import default_timer as timer

import numpy as np

from hapod.hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth
from hapod.mpi import MPIWrapper
from hapod.cellmodel.wrapper import CellModelSolver, CellModelPfieldProductOperator, CellModelOfieldProductOperator, CellModelStokesProductOperator, calculate_cellmodel_errors, create_and_scatter_cellmodel_parameters, get_num_chunks_and_num_timesteps


class HapodResult:
    pass


def cellmodel_binary_tree_hapod(testcase,
                                t_end,
                                dt,
                                grid_size_x,
                                grid_size_y,
                                desired_chunk_size,
                                tol,
                                eval_tol=None,
                                omega=0.95,
                                logfile=None,
                                incremental_gramian=True,
                                orthonormalize=True,
                                calc_eval_basis=False,
                                linear=True):

    # get MPI communicators
    mpi = MPIWrapper()

    # get cellmodel solver to create snapshots
    mu = create_and_scatter_cellmodel_parameters(mpi.comm_world)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    num_chunks, _ = get_num_chunks_and_num_timesteps(t_end, dt, desired_chunk_size)
    nc = solver.num_cells

    # calculate rooted tree depth
    node_binary_tree_depth = binary_tree_depth(mpi.comm_rank_0_group)
    node_binary_tree_depth = mpi.comm_proc.bcast(node_binary_tree_depth, root=0)
    rooted_tree_depth = num_chunks + node_binary_tree_depth

    # store HAPOD parameters for easier handling
    hapod_params = [
        HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega),
        HapodParameters(rooted_tree_depth, epsilon_ast=eval_tol, omega=omega)
    ] if calc_eval_basis else [HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega)]

    ret = [HapodResult() for _ in range(2)] if calc_eval_basis else [HapodResult()]
    products = [
        CellModelPfieldProductOperator(solver),
        CellModelOfieldProductOperator(solver),
        CellModelStokesProductOperator(solver)
    ]

    for r in ret:
        r.modes, r.max_vectors_before_pod, r.max_local_modes, r.total_num_snapshots, r.svals = [[0] * (2 * nc + 1) for i in range(5)]
    for i in range(num_chunks):
        ret[0].next_vecs = solver.next_n_timesteps(desired_chunk_size, dt)
        actual_chunk_size = len(ret[0].next_vecs[0])
        assert i == num_chunks - 1 or actual_chunk_size == desired_chunk_size
        chunk_sizes = [actual_chunk_size]
        if calc_eval_basis:
            ret[1].next_vecs = ret[0].next_vecs[1:] - r[0].next_vecs[:-1]
            chunk_sizes.append(len(ret[1].next_vecs[0]))

         # calculate POD of timestep vectors on each core
         # reuse storage of next_vecs to save memory
        for r, chunk_size, params in zip(ret, chunk_sizes, hapod_params):
            for k in range(2 * nc + 1):
                product = products[0 if k < nc else 1 if k < 2 * nc else 2]
                r.next_vecs[k], next_svals = local_pod([r.next_vecs[k]],
                                                     chunk_size,
                                                     params,
                                                     incremental_gramian=False,
                                                     product=product,
                                                     orthonormalize=orthonormalize)
                r.next_vecs[k].scal(next_svals)
                r.next_vecs[k], _, num_snaps_in_this_chunk, _ = mpi.comm_proc.gather_on_rank_0(r.next_vecs[k], chunk_size, num_modes_equal=False)

                # if there are already modes from the last chunk of vectors, perform another pod on rank 0
                if mpi.rank_proc == 0:
                    r.total_num_snapshots[k] += num_snaps_in_this_chunk
                    if i == 0:
                        r.max_vectors_before_pod[k] = len(r.next_vecs[k])
                        r.modes[k], r.svals[k] = local_pod([r.next_vecs[k]],
                                                     num_snaps_in_this_chunk,
                                                     params,
                                                     incremental_gramian=incremental_gramian,
                                                     product=product,
                                                     orthonormalize=orthonormalize)
                    else:
                        r.max_vectors_before_pod[k] = max(r.max_vectors_before_pod[k],
                                                       len(r.modes[k]) + len(r.next_vecs[k]))
                        r.modes[k], r.svals[k] = local_pod([[r.modes[k], r.svals[k]], r.next_vecs[k]],
                                                     r.total_num_snapshots[k],
                                                     params,
                                                     orthonormalize=orthonormalize,
                                                     incremental_gramian=incremental_gramian,
                                                     product=product,
                                                     root_of_tree=(i == num_chunks - 1 and mpi.size_rank_0_group == 1))
                    r.max_local_modes[k] = max(r.max_local_modes[k], len(r.modes[k]))
                    r.next_vecs[k] = None # to save memory

    # Finally, perform a HAPOD over a binary tree of nodes
    start2 = timer()
    for r, params in zip(ret, hapod_params):
        for k in range(2 * nc + 1):
            if mpi.rank_proc == 0:
                r.modes[k], r.svals[k], r.total_num_snapshots[k], max_vectors_before_pod_in_hapod, max_local_modes_in_hapod \
                    = binary_tree_hapod_over_ranks(mpi.comm_rank_0_group,
                                                   r.modes[k],
                                                   r.total_num_snapshots[k],
                                                   params,
                                                   svals=r.svals[k],
                                                   last_hapod=True,
                                                   incremental_gramian=incremental_gramian,
                                                   product=product,
                                                   orthonormalize=orthonormalize)
                r.max_vectors_before_pod[k] = max(r.max_vectors_before_pod[k], max_vectors_before_pod_in_hapod)
                r.max_local_modes[k] = max(r.max_local_modes[k], max_local_modes_in_hapod)
            else:
                r.modes[k], r.svals[k], r.total_num_snapshots[k] = (np.empty(shape=(0, 0)), None, None)

            # calculate max number of local modes
            gathered_max_vectors_before_pod = mpi.comm_world.gather(r.max_vectors_before_pod[k], root=0)
            gathered_max_local_modes = mpi.comm_world.gather(r.max_local_modes[k], root=0)
            if mpi.rank_world == 0:
                r.max_vectors_before_pod[k] = max(gathered_max_vectors_before_pod)
                r.max_local_modes[k] = max(gathered_max_local_modes)
                # r.final_hapod_time[k] = timer() - start2
                # r.time[k] = timer() - start

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        tps = ["snapshots", "nonlinear evaluations"] if calc_eval_basis else ["snapshots"]
        for r, tp in zip(ret, tps):
            for k in range(2 * nc + 1):
                eq_name = "pfield" if k < nc else "ofield" if k < 2 * nc else "stokes"
                logfile.write("The {} HAPOD resulted in {} final modes taken from a total of {} {}!\n".format(
                    eq_name, len(r.modes[k]), r.total_num_snapshots[k], tp))
                logfile.write("The maximal number of local modes was: {}\n".format(r.max_local_modes[k]))
                logfile.write("The maximal number of input vectors to a local POD was: {}\n".format(
                    r.max_vectors_before_pod[k]))
                # logfile.write("Time for final HAPOD over nodes: {}\n".format(r.final_hapod_time[k]))
                # logfile.write("Time for all: {}\n".format(s.time))
        logfile.write("The maximum amount of memory used on rank 0 was: {} GB\n".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2))

    return ret, mu, mpi, solver


if __name__ == "__main__":
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    tol = 1e-4 if argc < 7 else float(sys.argv[6])
    chunk_size = 5 if argc < 8 else int(sys.argv[7])
    omega = 0.95 if argc < 9 else float(sys.argv[8])
    inc_gramian = True if argc < 10 else not (sys.argv[9] == "False" or sys.argv[9] == "0")
    filename = "cellmodel_binary_tree_hapod_grid_%dx%d_chunksize_%d_tol_%f_omega_%f.log" % (grid_size_x, grid_size_y,
                                                                                        chunk_size, tol, omega)
    logfile = open(filename, 'w')
    ret, mu, mpi, _ = cellmodel_binary_tree_hapod(
        testcase,
        t_end,
        dt,
        grid_size_x,
        grid_size_y,
        chunk_size,
        tol=tol,
        eval_tol=tol,
        omega=omega,
        logfile=logfile,
        incremental_gramian=True,
        orthonormalize=True,
        calc_eval_basis=False,
        linear=True)

    # Broadcast modes to all ranks, if possible via shared memory
    retlistvecs = True
    for r in ret:
      r.wins = [None] * len(r.modes)
      for k in range(len(r.modes)):
        r.modes[k], r.wins[k] = mpi.shared_memory_bcast_modes(r.modes[k], returnlistvectorarray=retlistvecs)

    # calculate errors
    errors = calculate_cellmodel_errors(ret[0].modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu, mpi, logfile=logfile)

    # free MPI windows
    for r in ret:
        for win in r.wins:
            win.Free()
    mpi.comm_world.Barrier()

    if mpi.rank_world == 0:
        with open(filename, "r") as logfile:
            print("\n\n\nResults:\n")
            print(logfile.read())
