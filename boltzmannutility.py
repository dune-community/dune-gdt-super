import math
import random
from timeit import default_timer as timer

import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace

from boltzmann import wrapper
from libhapodgdt import CellModelSolver

def create_and_scatter_boltzmann_parameters(comm, min_param=0., max_param=8.):
    ''' Samples all 3 parameters uniformly with the same width and adds random parameter combinations until
        comm.Get_size() parameters are created. After that, parameter combinations are scattered to ranks. '''
    num_samples_per_parameter = int(comm.Get_size()**(1. / 3.) + 0.1)
    sample_width = (max_param - min_param) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sigma_s_scattering_range = sigma_s_absorbing_range = sigma_a_absorbing_range = np.arange(
        min_param, max_param + 1e-13, sample_width)
    sigma_a_scattering_range = [0.]
    parameters_list = []
    for sigma_s_scattering in sigma_s_scattering_range:
        for sigma_s_absorbing in sigma_s_absorbing_range:
            for sigma_a_scattering in sigma_a_scattering_range:
                for sigma_a_absorbing in sigma_a_absorbing_range:
                    parameters_list.append(
                        [sigma_s_scattering, sigma_s_absorbing, sigma_a_scattering, sigma_a_absorbing])
    while len(parameters_list) < comm.Get_size():
        parameters_list.append([
            random.uniform(min_param, max_param),
            random.uniform(min_param, max_param), 0.,
            random.uniform(min_param, max_param)
        ])
    return comm.scatter(parameters_list, root=0)


def create_boltzmann_solver(gridsize, mu, linear=True):
    return wrapper.Solver(
        "boltzmann_sigma_s_s_" + str(mu[0]) + "_a_" + str(mu[1]) + "sigma_t_s_" + str(mu[2]) + "_a_" + str(mu[3]),
        2000000, gridsize, False, False, *mu, linear)


def solver_statistics(solver, chunk_size, with_half_steps=True):
    num_time_steps = math.ceil(solver.t_end() / solver.time_step_length()) + 1.
    if with_half_steps:
        num_time_steps += num_time_steps - 1
    num_chunks = int(math.ceil(num_time_steps / chunk_size))
    last_chunk_size = num_time_steps - chunk_size * (num_chunks - 1)
    assert num_chunks >= 2
    assert 1 <= last_chunk_size <= chunk_size
    return num_chunks, num_time_steps


def calculate_trajectory_error(final_modes, grid_size, mu, with_half_steps=True):
    error = 0
    solver = create_boltzmann_solver(grid_size, mu)
    while not solver.finished():
        next_vectors = solver.next_n_timesteps(1, with_half_steps)
        next_vectors_np = NumpyVectorSpace(next_vectors[0].dim).from_data(next_vectors.data)
        error += np.sum((next_vectors_np - final_modes.lincomb(next_vectors_np.dot(final_modes))).l2_norm()**2)
    return error


def calculate_total_projection_error(final_modes,
                                     grid_size,
                                     mu,
                                     total_num_snapshots,
                                     mpi_wrapper,
                                     with_half_steps=True):
    trajectory_error = calculate_trajectory_error(final_modes, grid_size, mu, with_half_steps)
    trajectory_errors = mpi_wrapper.comm_world.gather(trajectory_error, root=0)
    error = 0
    if mpi_wrapper.rank_world == 0:
        error = np.sqrt(np.sum(trajectory_errors) / total_num_snapshots)
    return error


def calculate_error(final_modes, grid_size, mu, total_num_snapshots, mpi_wrapper, with_half_steps=True, logfile=None):
    ''' Calculates projection error. As we cannot store all snapshots due to memory restrictions, the
        problem is solved again and the error calculated on the fly'''
    start = timer()
    err = calculate_total_projection_error(final_modes, grid_size, mu, total_num_snapshots, mpi_wrapper,
                                           with_half_steps)
    err = err if grid_size == 0 else err / grid_size
    elapsed = timer() - start
    if mpi_wrapper.rank_world == 0 and logfile is not None:
        logfile.write("Time used for calculating error: " + str(elapsed) + "\n")
        logfile.write("l2_error is: " + str(err) + "\n")
        logfile.close()
    return err