from numbers import Number
from timeit import default_timer as timer
import random
import resource
import sys
import numpy as np
from statistics import mean

from hapod.cellmodel.wrapper import CellModelSolver, DuneCellModel, CellModel, CellModel, ProjectedSystemOperator, CellModelReductor
from hapod.mpi import MPIWrapper
from hapod.boltzmann.utility import solver_statistics
from hapod.hapod import local_pod, HapodParameters, binary_tree_hapod_over_ranks, binary_tree_depth

# set_log_levels({'pymor.algorithms.newton': 'WARN'})


def binary_tree_hapod(cellmodel,
                      chunk_size,
                      tol,
                      mpi,
                      reduce_pfield,
                      reduce_ofield,
                      reduce_stokes,
                      include_newton_stages,
                      omega=0.95,
                      incremental_gramian=True,
                      orthonormalize=True):

    start = timer()

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    num_chunks, num_timesteps = solver_statistics(cellmodel, chunk_size, with_half_steps=False)
    dt = cellmodel.dt

    # calculate rooted tree depth
    node_binary_tree_depth = binary_tree_depth(mpi.comm_rank_0_group)
    node_binary_tree_depth = mpi.comm_proc.bcast(node_binary_tree_depth, root=0)
    rooted_tree_depth = num_chunks + node_binary_tree_depth

    # store HAPOD parameters for easier handling
    hapod_params = HapodParameters(rooted_tree_depth, epsilon_ast=tol, omega=omega)

    indices = []
    if reduce_pfield:
        indices.append(0)
    if reduce_ofield:
        indices.append(1)
    if reduce_stokes:
        indices.append(2)
    U = cellmodel.solution_space.empty()
    modes = [None] * 3
    svals = [None] * 3
    max_vectors_before_pod, max_local_modes, total_num_snapshots, svals = [[0] * 3, [0] * 3, [0] * 3, [[]] * 3]
    timestep_vec = cellmodel.initial_values
    U.append(timestep_vec)
    t = 0.
    for i in range(num_chunks):
        new_vecs = [U._blocks[0].empty(), U._blocks[1].empty(), U._blocks[2].empty()]
        if i == 0:     # add initial values
            for k in indices:
                new_vecs[k].append(timestep_vec._blocks[k])
        for j in range(chunk_size - 1 if i == 0 else chunk_size):     # if i == 0, initial values are already in chunk
            if include_newton_stages:
                timestep_vec, t, timestep_stages = cellmodel.next_time_step(
                    timestep_vec, t, mu=mu, return_stages=True)
            else:
                timestep_vec, t = cellmodel.next_time_step(timestep_vec, t, mu=mu, return_stages=False)
            if timestep_vec is None:
                assert i == num_chunks - 1
                assert j != 0
                break
            U.append(timestep_vec)
            for k in indices:
                new_vecs[k].append(timestep_vec._blocks[k])
                if include_newton_stages:
                    new_vecs[k].append(timestep_stages[k])

        #      num_snapshots = (len(new_pfield_vecs), len(new_ofield_vecs), len(new_stokes_vecs))
        #      if not include_newton_stages:
        #          assert num_snapshots > 0 and (num_snapshots == chunk_size or i == num_chunks - 1)
        #      else
        #          assert num_snapshots > 0

        # calculate POD of timestep vectors on each core
        num_snapshots_in_this_chunk = [None] * 3
        gathered_modes = [None] * 3
        for k in indices:
            local_vecs = new_vecs[k]
            local_vecs, local_svals = local_pod([local_vecs],
                                                len(local_vecs),
                                                hapod_params,
                                                incremental_gramian=False,
                                                orthonormalize=orthonormalize)
            local_vecs.scal(local_svals)
            gathered_modes[k], _, num_snapshots_in_this_chunk[k], _ = mpi.comm_proc.gather_on_rank_0(
                local_vecs, len(local_vecs), num_modes_equal=False)
        del local_vecs, new_vecs

        # if there are already modes from the last chunk of vectors, perform another pod on rank 0
        if mpi.rank_proc == 0:
            for k in indices:
                total_num_snapshots[k] += num_snapshots_in_this_chunk[k]
                if i == 0:
                    modes[k], svals[k] = local_pod([gathered_modes[k]],
                                                   num_snapshots_in_this_chunk[k],
                                                   hapod_params,
                                                   orthonormalize=orthonormalize)
                else:
                    max_vectors_before_pod[k] = max(max_vectors_before_pod[k], len(modes[k]) + len(gathered_modes[k]))
                    modes[k], svals[k] = local_pod([[modes[k], svals[k]], gathered_modes[k]],
                                                   total_num_snapshots[k],
                                                   hapod_params,
                                                   orthonormalize=orthonormalize,
                                                   incremental_gramian=incremental_gramian,
                                                   root_of_tree=(i == num_chunks - 1 and mpi.size_rank_0_group == 1))
                max_local_modes[k] = max(max_local_modes[k], len(modes[k]))
            del gathered_modes

    # Finally, perform a HAPOD over a binary tree of nodes
    start2 = timer()
    for k in indices:
        if mpi.rank_proc == 0:
            modes[k], svals[k], total_num_snapshots[k], max_vectors_before_pod_in_hapod, max_local_modes_in_hapod \
                = binary_tree_hapod_over_ranks(mpi.comm_rank_0_group,
                                               modes[k],
                                               total_num_snapshots[k],
                                               hapod_params,
                                               svals=svals[k],
                                               last_hapod=True,
                                               incremental_gramian=incremental_gramian,
                                               orthonormalize=orthonormalize)
            max_vectors_before_pod[k] = max(max_vectors_before_pod[k], max_vectors_before_pod_in_hapod)
            max_local_modes[k] = max(max_local_modes[k], max_local_modes_in_hapod)
        else:
            modes[k], svals[k], total_num_snapshots[k] = (np.empty(shape=(0, 0)), None, None)

        # calculate max number of local modes
        max_vectors_before_pod[k] = mpi.comm_world.gather(max_vectors_before_pod[k], root=0)
        max_local_modes[k] = mpi.comm_world.gather(max_local_modes[k], root=0)
        if mpi.rank_world == 0:
            max_vectors_before_pod[k] = max(max_vectors_before_pod[k])
            max_local_modes[k] = max(max_local_modes[k])

    # write statistics to file
    if mpi.rank_world == 0:
        for k in indices:
            print(f"Hapod for index {k}")
            print("The HAPOD resulted in %d final modes taken from a total of %d snapshots!" % (len(
                modes[k]), total_num_snapshots[k]))
            print("The maximal number of local modes was: " + str(max_local_modes[k]))
            print("The maximal number of input vectors to a local POD was: " + str(max_vectors_before_pod[k]))
        print("The maximum amount of memory used on rank 0 was: " +
              str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB")
        print("Time for final HAPOD over nodes:" + str(timer() - start2))
        print("Time for all:" + str(timer() - start))

    return U, modes, svals, total_num_snapshots, max_vectors_before_pod, max_local_modes


if __name__ == "__main__":
    mpi = MPIWrapper()
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 30 if argc < 5 else int(sys.argv[4])
    grid_size_y = 30 if argc < 6 else int(sys.argv[5])
    visualize = True if argc < 7 else bool(sys.argv[6])
    subsampling = True if argc < 8 else bool(sys.argv[7])
    pol_order = 2
    chunk_size = 10
    pod_rtol = 1e-13
    basis_size_step = 50
    visualize_step = 50
    include_newton_stages = False
    reduce_pfield = True
    reduce_ofield = True
    reduce_stokes = True
    tested_param = 'Ca'
    Ca0 = 0.1
    Be0 = 0.3
    Pa0 = 1.0
    mus = []
    num_train_params = mpi.size_world
    rf = np.sqrt(10)
    factors = np.array([(rf**2)**(i / (num_train_params - 1)) / rf for i in range(num_train_params)])
    if tested_param == 'Ca':
        for Ca in factors * Ca0:
            mus.append({'Pa': Pa0, 'Be': Be0, 'Ca': Ca})
    elif tested_param == 'Be':
        for Be in factors * Be0:
            mus.append({'Pa': Pa0, 'Be': Be, 'Ca': Ca0})
    elif tested_param == 'Pa':
        for Pa in factors * Pa0:
            mus.append({'Pa': Pa, 'Be': Be0, 'Ca': Ca0})
    else:
        raise NotImplementedError(f"Wrong value of tested_param: {tested_param}")
    new_mus = []
    for i in range(len(mus)):
        new_mu = {'Pa': Pa0, 'Be': Be0, 'Ca': Ca0}
        new_mu[tested_param] = random.uniform(new_mu[tested_param] / rf, new_mu[tested_param] * rf)
        new_mus.append(new_mu)
    print(mus)
    print(new_mus)
    mu = mpi.comm_world.scatter(mus, root=0)
    new_mu = mpi.comm_world.scatter(new_mus, root=0)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, pol_order, mu)
    num_cells = solver.num_cells
    m = DuneCellModel(solver, dt, t_end)
    U, modes, svals, _, _, _ = binary_tree_hapod(
        m, chunk_size, pod_rtol, mpi, reduce_pfield, reduce_ofield, reduce_stokes, include_newton_stages, omega=0.95)
    full_pfield_basis, full_ofield_basis, full_stokes_basis = modes
    if reduce_pfield:
        full_pfield_basis, win_pfield = mpi.shared_memory_bcast_modes(full_pfield_basis, returnlistvectorarray=True)
    if reduce_ofield:
        full_ofield_basis, win_ofield = mpi.shared_memory_bcast_modes(full_ofield_basis, returnlistvectorarray=True)
    if reduce_stokes:
        full_stokes_basis, win_stokes = mpi.shared_memory_bcast_modes(full_stokes_basis, returnlistvectorarray=True)

    filename = "rom_results_least_squares_grid{}x{}_tend{}_{}_param{}_pfield_{}_ofield_{}_stokes_{}.txt".format(
        grid_size_x, grid_size_y, t_end, 'snapsandstages' if include_newton_stages else 'snaps', tested_param,
        'pod' if reduce_pfield else 'none', 'pod' if reduce_ofield else 'none', 'pod' if reduce_stokes else 'none')

    if mpi.rank_world == 0:
        with open(filename, 'w') as ff:
            ff.write(
                f"{filename}\nTrained with {len(mus)} Parameters for {tested_param}: {[param[tested_param] for param in mus]}\n"
                f"Tested with {len(new_mus)} new Parameters: {[param[tested_param] for param in new_mus]}\n")
            ff.write("num_basis_vecs pfield_trained ofield_trained stokes_trained pfield_new ofield_new stokes_new\n")

    # solve full-order model for new param
    U_new_mu = m.solve(mu=new_mu, return_stages=False)
    Be, Ca, Pa = (float(new_mu['Be']), float(new_mu['Ca']), float(new_mu['Pa']))
    m.visualize(U_new_mu, prefix=f"fullorder_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling, every_nth=visualize_step)

    min_basis_len = basis_size_step
    pfield_basis_max_len = len(full_pfield_basis) if reduce_pfield else np.inf
    ofield_basis_max_len = len(full_ofield_basis) if reduce_ofield else np.inf
    stokes_basis_max_len = len(full_stokes_basis) if reduce_stokes else np.inf
    max_basis_len = min(pfield_basis_max_len, ofield_basis_max_len, stokes_basis_max_len)

    for basis_len in range(min_basis_len, max_basis_len, basis_size_step):

        pfield_basis = full_pfield_basis[:basis_len] if reduce_pfield else None
        ofield_basis = full_ofield_basis[:basis_len] if reduce_ofield else None
        stokes_basis = full_stokes_basis[:basis_len] if reduce_stokes else None

        reduced_prefix = "param{}_{}_pfield_{}_ofield_{}_stokes_{}".format(
            tested_param, 'snapsandstages' if include_newton_stages else 'snaps',
            f'pod{len(pfield_basis)}' if reduce_pfield else 'none',
            f'pod{len(ofield_basis)}' if reduce_ofield else 'none',
            f'pod{len(stokes_basis)}' if reduce_stokes else 'none')

        reductor = CellModelReductor(
            m,
            pfield_basis,
            ofield_basis,
            stokes_basis,
            least_squares_pfield=True if reduce_pfield else False,
            least_squares_ofield=True if reduce_ofield else False,
            least_squares_stokes=True if reduce_stokes else False)
        rom = reductor.reduce()

        ################## solve reduced model for trained parameters ####################
        u = rom.solve(mu, return_stages=False)
        U_rom = reductor.reconstruct(u)

        ################## test new parameters #######################
        # solve reduced model for new params
        u_new_mu = rom.solve(new_mu, return_stages=False)
        U_rom_new_mu = reductor.reconstruct(u_new_mu)
        Be, Ca, Pa = (float(new_mu['Be']), float(new_mu['Ca']), float(new_mu['Pa']))
        m.visualize(
            U_rom_new_mu,
            prefix=f"{reduced_prefix}_Be{Be}_Ca{Ca}_Pa{Pa}",
            subsampling=subsampling,
            every_nth=visualize_step)

        # calculate_errors
        pfield_rel_errors = (U._blocks[0] - U_rom._blocks[0]).norm() / U._blocks[0].norm()
        ofield_rel_errors = (U._blocks[1] - U_rom._blocks[1]).norm() / U._blocks[1].norm()
        stokes_rel_errors = (U._blocks[2] - U_rom._blocks[2]).norm() / U._blocks[2].norm()
        pfield_rel_errors_new_mu = (U_new_mu._blocks[0] - U_rom_new_mu._blocks[0]).norm() / U_new_mu._blocks[0].norm()
        ofield_rel_errors_new_mu = (U_new_mu._blocks[1] - U_rom_new_mu._blocks[1]).norm() / U_new_mu._blocks[1].norm()
        stokes_rel_errors_new_mu = (U_new_mu._blocks[2] - U_rom_new_mu._blocks[2]).norm() / U_new_mu._blocks[2].norm()
        pfield_rel_errors = mpi.comm_world.gather(pfield_rel_errors, root=0)
        ofield_rel_errors = mpi.comm_world.gather(ofield_rel_errors, root=0)
        stokes_rel_errors = mpi.comm_world.gather(stokes_rel_errors, root=0)
        pfield_rel_errors_new_mus = mpi.comm_world.gather(pfield_rel_errors_new_mu, root=0)
        ofield_rel_errors_new_mus = mpi.comm_world.gather(ofield_rel_errors_new_mu, root=0)
        stokes_rel_errors_new_mus = mpi.comm_world.gather(stokes_rel_errors_new_mu, root=0)

        if mpi.rank_world == 0:
            pfield_rel_errors = np.concatenate(pfield_rel_errors)
            ofield_rel_errors = np.concatenate(ofield_rel_errors)
            stokes_rel_errors = np.concatenate(stokes_rel_errors)
            pfield_rel_errors_new_mus = np.concatenate(pfield_rel_errors_new_mus)
            ofield_rel_errors_new_mus = np.concatenate(ofield_rel_errors_new_mus)
            stokes_rel_errors_new_mus = np.concatenate(stokes_rel_errors_new_mus)
            with open(filename, 'a') as ff:
                ff.write("{} {} {} {} {} {} {}\n".format(
                    basis_len, mean([err for err in pfield_rel_errors if not np.isnan(err)]),
                    mean([err for err in ofield_rel_errors if not np.isnan(err)]),
                    mean([err for err in stokes_rel_errors if not np.isnan(err)]),
                    mean([err for err in pfield_rel_errors_new_mus if not np.isnan(err)]),
                    mean([err for err in ofield_rel_errors_new_mus if not np.isnan(err)]),
                    mean([err for err in stokes_rel_errors_new_mus if not np.isnan(err)])))
