import resource
import sys
from timeit import default_timer as timer

import numpy as np

from hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth
from mpiwrapper import MPIWrapper
from boltzmann.wrapper import CellModelSolver, CellModelPfieldProductOperator, CellModelOfieldProductOperator, CellModelStokesProductOperator, calculate_cellmodel_error, create_and_scatter_cellmodel_parameters, get_num_chunks_and_num_timesteps


class HapodResult:
    pass


def cellmodel_binary_tree_hapod(testcase,
                                t_end,
                                dt,
                                grid_size_x,
                                grid_size_y,
                                chunk_size,
                                tol,
                                eval_tol=None,
                                omega=0.95,
                                logfile=None,
                                incremental_gramian=True,
                                orthonormalize=True,
                                calc_eval_basis=False,
                                linear=True):

    start = timer()

    # get MPI communicators
    mpi = MPIWrapper()

    # get cellmodel solver to create snapshots
    mu = create_and_scatter_cellmodel_parameters(mpi.comm_world)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    num_chunks, _ = get_num_chunks_and_num_timesteps(t_end, dt, chunk_size)

    # calculate rooted tree depth
    node_binary_tree_depth = binary_tree_depth(mpi.comm_rank_0_group)
    node_binary_tree_depth = mpi.comm_proc.bcast(node_binary_tree_depth, root=0)
    rooted_tree_depth = num_chunks + node_binary_tree_depth

    # store HAPOD parameters for easier handling
    hapod_params = [
        HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega),
        HapodParameters(rooted_tree_depth, epsilon_ast=eval_tol, omega=omega)
    ] if calc_eval_basis else [HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega)]

    # Outer list has entries for pfield, ofield, stokes, respectively. Each entry is a list of length 2, first one is for snapshots and and second one for non-linear evaluations
    ret = [[HapodResult() for _ in range(2)] if calc_eval_basis else [HapodResult()] for _ in range(3)]
    products = [
        CellModelPfieldProductOperator(solver),
        CellModelOfieldProductOperator(solver),
        CellModelStokesProductOperator(solver)
    ]

    ret[0][0].modes = solver.pfield_solution_space.empty()
    ret[1][0].modes = solver.ofield_solution_space.empty()
    ret[2][0].modes = solver.stokes_solution_space.empty()
    for r in ret:
        if calc_eval_basis:
            r[1].modes = r[0].modes.copy()
        for s in r:
            s.max_vectors_before_pod, s.max_local_modes, s.total_num_snapshots, s.svals = [0, 0, 0, []]
    for i in range(num_chunks):
        ret[0][0].next_vecs, ret[1][0].next_vecs, ret[2][0].next_vecs = solver.next_n_timesteps(chunk_size, dt)
        assert len(ret[0][0].next_vecs) == len(ret[1][0].next_vecs) == len(ret[2][0].next_vecs)
        assert i == num_chunks - 1 or len(ret[0][0].next_vecs) == chunk_size
        for r, product in zip(ret, products):
            r[0].num_vecs = len(r[0].next_vecs)
            if calc_eval_basis:
                r[1].next_vecs = r[0].next_vecs[1:] - r[0].next_vecs[:-1]
                r[1].num_vecs = len(r[1].next_vecs)

            # calculate POD of timestep vectors on each core
            # reuse storage of next_vecs to save memory
            for s, params in zip(r, hapod_params):
                s.next_vecs, s.next_svals = local_pod([s.next_vecs],
                                                      s.num_vecs,
                                                      params,
                                                      incremental_gramian=False,
                                                      product=product,
                                                      orthonormalize=orthonormalize)
                s.next_vecs.scal(s.next_svals)
                s.gathered_vectors, _, s.num_snaps_in_this_chunk, _ = mpi.comm_proc.gather_on_rank_0(
                    s.next_vecs, s.num_vecs, num_modes_equal=False)
                del s.next_vecs

                # if there are already modes from the last chunk of vectors, perform another pod on rank 0
                if mpi.rank_proc == 0:
                    s.total_num_snapshots += s.num_snaps_in_this_chunk
                    if i == 0:
                        s.max_vectors_before_pod = len(s.gathered_vectors)
                        s.modes, s.svals = local_pod([s.gathered_vectors],
                                                     s.num_snaps_in_this_chunk,
                                                     params,
                                                     incremental_gramian=incremental_gramian,
                                                     product=product,
                                                     orthonormalize=orthonormalize)
                    else:
                        s.max_vectors_before_pod = max(s.max_vectors_before_pod,
                                                       len(s.modes) + len(s.gathered_vectors))
                        s.modes, s.svals = local_pod([[s.modes, s.svals], s.gathered_vectors],
                                                     s.total_num_snapshots,
                                                     params,
                                                     orthonormalize=orthonormalize,
                                                     incremental_gramian=incremental_gramian,
                                                     product=product,
                                                     root_of_tree=(i == num_chunks - 1 and mpi.size_rank_0_group == 1))
                    s.max_local_modes = max(s.max_local_modes, len(s.modes))
                    del s.gathered_vectors

                # Finally, perform a HAPOD over a binary tree of nodes
                start2 = timer()
                if mpi.rank_proc == 0:
                    s.modes, s.svals, s.total_num_snapshots, s.max_vectors_before_pod_in_hapod, s.max_local_modes_in_hapod \
                        = binary_tree_hapod_over_ranks(mpi.comm_rank_0_group,
                                                       s.modes,
                                                       s.total_num_snapshots,
                                                       params,
                                                       svals=s.svals,
                                                       last_hapod=True,
                                                       incremental_gramian=incremental_gramian,
                                                       product=product,
                                                       orthonormalize=orthonormalize)
                    s.max_vectors_before_pod = max(s.max_vectors_before_pod, s.max_vectors_before_pod_in_hapod)
                    s.max_local_modes = max(s.max_local_modes, s.max_local_modes_in_hapod)

                else:
                    s.modes, s.svals, s.total_num_snapshots = (np.empty(shape=(0, 0)), None, None)

                # calculate max number of local modes
                gathered_max_vectors_before_pod = mpi.comm_world.gather(s.max_vectors_before_pod, root=0)
                gathered_max_local_modes = mpi.comm_world.gather(s.max_local_modes, root=0)
                if mpi.rank_world == 0:
                    s.max_vectors_before_pod = max(gathered_max_vectors_before_pod)
                    s.max_local_modes = max(gathered_max_local_modes)
                    s.final_hapod_time = timer() - start2
                    s.time = timer() - start

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        tps = ["snapshots", "nonlinear evaluations"] if calc_eval_basis else ["snapshots"]
        for r, eq_name in zip(ret, ["pfield", "ofield", "stokes"]):
            for s, tp in zip(r, tps):
                logfile.write("The {} HAPOD resulted in {} final modes taken from a total of {} {}!\n".format(
                    eq_name, len(s.modes), s.total_num_snapshots, tp))
                logfile.write("The maximal number of local modes was: {}\n".format(s.max_local_modes))
                logfile.write("The maximal number of input vectors to a local POD was: {}\n".format(
                    s.max_vectors_before_pod))
                logfile.write("Time for final HAPOD over nodes: {}\n".format(s.final_hapod_time))
                logfile.write("Time for all: {}\n".format(s.time))
        logfile.write("The maximum amount of memory used on rank 0 was: {} GB\n".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2))

    return ret, mu, mpi


if __name__ == "__main__":
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    tol = 1e-4 if argc < 7 else float(sys.argv[6])
    chunk_size = 5 if argc < 8 else int(sys.argv[7])
    omega = 0.95 if argc < 9 else  float(sys.argv[8])
    inc_gramian = True if argc < 10 else not (sys.argv[9] == "False" or sys.argv[9] == "0")
    filename = "cellmodel_binary_tree_hapod_grid_%dx%d_chunksize_%d_tol_%f_omega_%f" % (grid_size_x, grid_size_y,
                                                                                        chunk_size, tol, omega)
    logfile = open(filename, 'w')
    ret, mu, mpi = cellmodel_binary_tree_hapod(testcase,
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
    final_modes_pfield, win_pfield = mpi.shared_memory_bcast_modes(ret[0][0].modes, True)
    final_modes_ofield, win_ofield = mpi.shared_memory_bcast_modes(ret[1][0].modes, True)
    final_modes_stokes, win_stokes = mpi.shared_memory_bcast_modes(ret[2][0].modes, True)
    err_pfield, err_ofield, err_stokes = calculate_cellmodel_error(final_modes_pfield,
                                                                   final_modes_ofield,
                                                                   final_modes_stokes,
                                                                   testcase,
                                                                   t_end,
                                                                   dt,
                                                                   grid_size_x,
                                                                   grid_size_y,
                                                                   mu,
                                                                   mpi,
                                                                   logfile=logfile)
    win_pfield.Free()
    win_ofield.Free()
    win_stokes.Free()
    logfile.close()
    mpi.comm_world.Barrier()
    if mpi.rank_world == 0:
        with open(filename, "r") as logfile:
            print("\n\n\nResults:\n")
            print(logfile.read())
