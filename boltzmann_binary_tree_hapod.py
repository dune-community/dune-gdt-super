import resource
import sys
from timeit import default_timer as timer

import numpy as np

from hapod.boltzmann.utility import (calculate_error, create_and_scatter_boltzmann_parameters, create_boltzmann_solver,
                                     solver_statistics)
from hapod.hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth
from hapod.mpi import MPIWrapper


def boltzmann_binary_tree_hapod(grid_size,
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

    # get boltzmann solver to create snapshots
    mu = create_and_scatter_boltzmann_parameters(mpi.comm_world)
    solver = create_boltzmann_solver(grid_size, mu, linear=linear)
    num_chunks, num_timesteps = solver_statistics(solver, chunk_size)
    dt = solver.impl.time_step_length()

    # calculate rooted tree depth
    node_binary_tree_depth = binary_tree_depth(mpi.comm_rank_0_group)
    node_binary_tree_depth = mpi.comm_proc.bcast(node_binary_tree_depth, root=0)
    rooted_tree_depth = num_chunks + node_binary_tree_depth

    # store HAPOD parameters for easier handling
    hapod_params = HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega)
    eval_hapod_params = HapodParameters(rooted_tree_depth, epsilon_ast=eval_tol, omega=omega)

    max_vectors_before_pod, max_local_modes, total_num_snapshots, svals = [0, 0, 0, []]
    max_eval_vectors_before_pod, max_local_eval_modes, total_num_evals, eval_svals = [0, 0, 0, []]
    modes = solver.solution_space.empty()
    eval_modes = solver.solution_space.empty()
    for i in range(num_chunks):
        assert chunk_size % 2 == 0
        timestep_vectors = solver.next_n_timesteps(chunk_size)
        # assert (len(timestep_vectors) % 2 == 1) == (i == 0), (len(timestep_vectors), i)
        num_snapshots = len(timestep_vectors)
        assert i + 1 == num_chunks or num_snapshots == chunk_size
        if calc_eval_basis:
            if len(timestep_vectors) % 2 == 0:
                lf_eval_vectors = timestep_vectors[1::2] - timestep_vectors[::2]
            else:
                # can only happen in last chunk
                lf_eval_vectors = timestep_vectors[1::2] - timestep_vectors[:-1:2]
            num_evals = len(lf_eval_vectors)

        # calculate POD of timestep vectors on each core
        timestep_vectors, timestep_svals = local_pod([timestep_vectors],
                                                     num_snapshots,
                                                     hapod_params,
                                                     incremental_gramian=False,
                                                     orthonormalize=orthonormalize)
        timestep_vectors.scal(timestep_svals)
        gathered_vectors, _, num_snapshots_in_this_chunk, _ = mpi.comm_proc.gather_on_rank_0(
            timestep_vectors, num_snapshots, num_modes_equal=False)
        del timestep_vectors

        # if there are already modes from the last chunk of vectors, perform another pod on rank 0
        if mpi.rank_proc == 0:
            total_num_snapshots += num_snapshots_in_this_chunk
            if i == 0:
                modes, svals = local_pod([gathered_vectors],
                                         num_snapshots_in_this_chunk,
                                         hapod_params,
                                         orthonormalize=orthonormalize)
            else:
                max_vectors_before_pod = max(max_vectors_before_pod, len(modes) + len(gathered_vectors))
                modes, svals = local_pod([[modes, svals], gathered_vectors],
                                         total_num_snapshots,
                                         hapod_params,
                                         orthonormalize=orthonormalize,
                                         incremental_gramian=incremental_gramian,
                                         root_of_tree=(i == num_chunks - 1 and mpi.size_rank_0_group == 1))
            max_local_modes = max(max_local_modes, len(modes))
            del gathered_vectors

        if calc_eval_basis:
            lf_eval_vectors, lf_eval_svals = local_pod([lf_eval_vectors],
                                                       num_evals,
                                                       eval_hapod_params,
                                                       incremental_gramian=False,
                                                       orthonormalize=orthonormalize)
            lf_eval_vectors.scal(lf_eval_svals)
            gathered_eval_vectors, _, num_evals_in_this_chunk, _ = mpi.comm_proc.gather_on_rank_0(
                lf_eval_vectors, num_evals, num_modes_equal=False)
            del lf_eval_vectors

            # if there are already modes from the last chunk of vectors, perform another pod on rank 0
            if mpi.rank_proc == 0:
                total_num_evals += num_evals_in_this_chunk
                if i == 0:
                    eval_modes, eval_svals = local_pod([gathered_eval_vectors],
                                                       num_evals_in_this_chunk,
                                                       eval_hapod_params,
                                                       orthonormalize=orthonormalize)
                else:
                    max_eval_vectors_before_pod = max(max_eval_vectors_before_pod,
                                                      len(eval_modes) + len(gathered_eval_vectors))
                    eval_modes, eval_svals = local_pod([[eval_modes, eval_svals], gathered_eval_vectors],
                                                       total_num_evals,
                                                       eval_hapod_params,
                                                       orthonormalize=orthonormalize,
                                                       incremental_gramian=incremental_gramian,
                                                       root_of_tree=(i == num_chunks - 1
                                                                     and mpi.size_rank_0_group == 1))
                max_local_eval_modes = max(max_local_eval_modes, len(eval_modes))
                del gathered_eval_vectors

    # Finally, perform a HAPOD over a binary tree of nodes
    start2 = timer()
    if mpi.rank_proc == 0:
        final_modes, svals, total_num_snapshots, max_vectors_before_pod_in_hapod, max_local_modes_in_hapod \
            = binary_tree_hapod_over_ranks(mpi.comm_rank_0_group,
                                           modes,
                                           total_num_snapshots,
                                           hapod_params,
                                           svals=svals,
                                           last_hapod=True,
                                           incremental_gramian=incremental_gramian,
                                           orthonormalize=orthonormalize)
        max_vectors_before_pod = max(max_vectors_before_pod, max_vectors_before_pod_in_hapod)
        max_local_modes = max(max_local_modes, max_local_modes_in_hapod)
        del modes

        if calc_eval_basis:
            final_eval_modes, eval_svals, total_num_evals, max_eval_vectors_before_pod_in_hapod, max_local_eval_modes_in_hapod \
                = binary_tree_hapod_over_ranks(mpi.comm_rank_0_group,
                                               eval_modes,
                                               total_num_evals,
                                               eval_hapod_params,
                                               svals=eval_svals,
                                               last_hapod=True,
                                               incremental_gramian=incremental_gramian,
                                               orthonormalize=orthonormalize)
            max_eval_vectors_before_pod = max(max_eval_vectors_before_pod, max_eval_vectors_before_pod_in_hapod)
            max_local_eval_modes = max(max_local_eval_modes, max_local_eval_modes_in_hapod)
            del eval_modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0, 0)), None, None)
        if calc_eval_basis:
            final_eval_modes, eval_svals, total_num_evals = (np.empty(shape=(0, 0)), None, None)

    # calculate max number of local modes
    max_vectors_before_pod = mpi.comm_world.gather(max_vectors_before_pod, root=0)
    max_local_modes = mpi.comm_world.gather(max_local_modes, root=0)
    if mpi.rank_world == 0:
        max_vectors_before_pod = max(max_vectors_before_pod)
        max_local_modes = max(max_local_modes)

    if calc_eval_basis:
        max_eval_vectors_before_pod = mpi.comm_world.gather(max_eval_vectors_before_pod, root=0)
        max_local_eval_modes = mpi.comm_world.gather(max_local_eval_modes, root=0)
        if mpi.rank_world == 0:
            max_eval_vectors_before_pod = max(max_eval_vectors_before_pod)
            max_local_eval_modes = max(max_local_eval_modes)

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("The HAPOD resulted in %d final modes taken from a total of %d snapshots!\n" %
                      (len(final_modes), total_num_snapshots))
        logfile.write("The maximal number of local modes was: " + str(max_local_modes) + "\n")
        logfile.write("The maximal number of input vectors to a local POD was: " + str(max_vectors_before_pod) + "\n")
        if calc_eval_basis:
            logfile.write("The DEIM HAPOD resulted in %d final modes taken from a total of %d snapshots!\n" %
                          (len(final_eval_modes), total_num_evals))
            logfile.write("The maximal number of local modes was: " + str(max_local_eval_modes) + "\n")
            logfile.write("The maximal number of input vectors to a local POD was: " +
                          str(max_eval_vectors_before_pod) + "\n")
        logfile.write("The maximum amount of memory used on rank 0 was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB\n")
        logfile.write("Time for final HAPOD over nodes:" + str(timer() - start2) + "\n")
        logfile.write("Time for all:" + str(timer() - start) + "\n")

    if not calc_eval_basis:
        final_eval_modes = eval_svals = total_num_evals = max_eval_vectors_before_pod = max_local_eval_modes = None
    return (final_modes, final_eval_modes, svals, eval_svals, total_num_snapshots, total_num_evals, mu, mpi,
            max_vectors_before_pod, max_eval_vectors_before_pod, max_local_modes, max_local_eval_modes, solver)


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 20 if argc < 2 else int(sys.argv[1])
    chunk_size = 6 if argc < 3 else int(sys.argv[2])
    tol = 1e-3 if argc < 4 else float(sys.argv[3])
    omega = 0.95 if argc < 5 else float(sys.argv[4])
    inc_gramian = True if argc < 6 else not (sys.argv[5] == "False" or sys.argv[5] == "0")
    filename = "HAPOD_binary_tree_gridsize_%d_chunksize_%d_tol_%f_omega_%f.log" % (grid_size, chunk_size, tol, omega)
    logfile = open(filename, "a")
    final_modes, _, _, _, total_num_snapshots, _, mu, mpi, _, _, _, _, _ = boltzmann_binary_tree_hapod(
        grid_size, chunk_size, tol * grid_size, None, omega=omega, logfile=logfile, incremental_gramian=inc_gramian)
    final_modes, win = mpi.shared_memory_bcast_modes(final_modes)
    calculate_error(final_modes, grid_size, mu, total_num_snapshots, mpi, logfile=logfile)
    win.Free()
    logfile.close()
    mpi.comm_world.Barrier()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
