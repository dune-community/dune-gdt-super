from timeit import default_timer as timer

import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace

from hapod.coordinatetransformedmn import wrapper

def create_coordinatetransformedmn_solver(gridsize, mu):
    prefix = "coordinatetransformedmn_sigma_s_s_{}_a_{}_sigma_t_s_{}_a_{}".format(*mu)
    return wrapper.Solver(prefix, 2000000, gridsize, False, False, *mu)

def calculate_trajectory_error(final_modes, grid_size, mu, with_half_steps=True):
    solver = create_coordinatetransformedmn_solver(grid_size, mu)
    vectors = solver.solve()
    vectors_np = NumpyVectorSpace(vectors[0].dim).from_data(vectors.data)
    error  = np.sum((vectors_np - final_modes.lincomb(vectors_np.dot(final_modes))).l2_norm()**2)
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
