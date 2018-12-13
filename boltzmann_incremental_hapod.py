import resource
import sys
from timeit import default_timer as timer

import numpy as np

from boltzmannutility import (calculate_error, create_and_scatter_boltzmann_parameters, create_boltzmann_solver,
                              solver_statistics)
from hapod import local_pod, HapodParameters, incremental_hapod_over_ranks
from mpiwrapper import MPIWrapper


def boltzmann_incremental_hapod(grid_size, chunk_size, tol, omega=0.95, logfile=None, incremental_gramian=True):

    start = timer()

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    mu = create_and_scatter_boltzmann_parameters(mpi.comm_world)
    solver = create_boltzmann_solver(grid_size, mu)
    num_chunks, num_timesteps = solver_statistics(solver, chunk_size)

    # calculate rooted tree depth
    rooted_tree_depth = num_chunks + mpi.size_rank_0_group

    # store HAPOD parameters for easier handling
    hapod_params = HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega)

    max_vectors_before_pod, max_local_modes, total_num_snapshots, svals = [0, 0, 0, []]
    modes = solver.solution_space.empty()
    for i in range(num_chunks):
        timestep_vectors = solver.next_n_time_steps(chunk_size)
        num_snapshots = len(timestep_vectors)
        # calculate POD of timestep vectors on each core
        timestep_vectors, timestep_svals = local_pod([timestep_vectors], num_snapshots, hapod_params,
                                                     incremental_gramian=False)
        timestep_vectors.scal(timestep_svals)
        gathered_vectors, _, num_snapshots_in_this_chunk, _ = mpi.comm_proc.gather_on_rank_0(timestep_vectors,
                                                                                             num_snapshots,
                                                                                             num_modes_equal=False)
        del timestep_vectors
        # if there are already modes from the last chunk of vectors, perform another pod on rank 0
        if mpi.rank_proc == 0:
            total_num_snapshots += num_snapshots_in_this_chunk
            if i == 0:
                modes, svals = local_pod([gathered_vectors], num_snapshots_in_this_chunk, hapod_params)
            else:
                max_vectors_before_pod = max(max_vectors_before_pod, len(modes) + len(gathered_vectors))
                modes, svals = local_pod([[modes, svals], gathered_vectors], total_num_snapshots,
                                         hapod_params, incremental_gramian=incremental_gramian,
                                         root_of_tree=(i == num_chunks-1 and mpi.size_rank_0_group == 1))
            max_local_modes = max(max_local_modes, len(modes))
            del gathered_vectors

    # Finally, perform a HAPOD over a binary tree of nodes
    start2 = timer()
    if mpi.rank_proc == 0:
        final_modes, svals, total_num_snapshots, max_vectors_before_pod_in_hapod, max_local_modes_in_hapod \
            = incremental_hapod_over_ranks(mpi.comm_rank_0_group,
                                           modes,
                                           total_num_snapshots,
                                           hapod_params,
                                           svals=svals,
                                           last_hapod=True,
                                           incremental_gramian=incremental_gramian)
        max_vectors_before_pod = max(max_vectors_before_pod, max_vectors_before_pod_in_hapod)
        max_local_modes = max(max_local_modes, max_local_modes_in_hapod)
        del modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0, 0)), None, None)

    # calculate max number of local modes
    max_vectors_before_pod = mpi.comm_world.gather(max_vectors_before_pod, root=0)
    max_local_modes = mpi.comm_world.gather(max_local_modes, root=0)
    if mpi.rank_world == 0:
        max_vectors_before_pod = max(max_vectors_before_pod)
        max_local_modes = max(max_local_modes)

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("The HAPOD resulted in %d final modes taken from a total of %d snapshots!\n"
                      % (len(final_modes), total_num_snapshots))
        logfile.write("The maximal number of local modes was: " + str(max_local_modes) + "\n")
        logfile.write("The maximal number of input vectors to a local POD was: " + str(max_vectors_before_pod) + "\n")
        logfile.write("The maximum amount of memory used on rank 0 was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        logfile.write("Time for final HAPOD over nodes:" + str(timer()-start2) + "\n")
        logfile.write("Time for all:" + str(timer()-start) + "\n")

    return final_modes, svals, total_num_snapshots, mu, mpi, max_vectors_before_pod, max_local_modes, solver


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    inc_gramian = not (sys.argv[5] == "False" or sys.argv[5] == "0") if len(sys.argv) > 5 else True
    filename = "boltzmann_incremental_hapod_gridsize_%d_chunksize_%d_tol_%f_omega_%f" \
               % (grid_size, chunk_size, tol, omega)
    logfile = open(filename, "a")
    final_modes, _, total_num_snapshots, mu, mpi, _, _, _ = boltzmann_incremental_hapod(grid_size, chunk_size,
                                                                                        tol * grid_size,
                                                                                        omega=omega, logfile=logfile,
                                                                                        incremental_gramian=inc_gramian)
    final_modes, win = mpi.shared_memory_bcast_modes(final_modes)
    calculate_error(final_modes, grid_size, mu, total_num_snapshots, mpi, grid_size, logfile=logfile)
    win.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
