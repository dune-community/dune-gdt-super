import math
import random
from timeit import default_timer as timer

import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace

from hapod.boltzmann.utility import create_and_scatter_boltzmann_parameters
from hapod.coordinatetransformedmn import wrapper


def convert_L2_l2(tol, grid_size, testcase, input_is_l2=False):
    r"""
    Calculates the l2 tolerance from a given L2 tolerance for a uniform finite volume discretization. If input_is_l2 is set to True,
    tol is assumed to be

    Parameters
    ----------
    tol: input tolerance in L^2 norm (l^2 norm if input_is_l2 == True)
    grid_size: number of grid elements per coordinate direction
    testcase: One of the testcases from AVAILABLE_TESTCASES (see hapod.coordinatetransformedmn.wrapper)
    input_is_l2: If set to True, this function takes a l^2 tolerance and returns the L^2 tolerance

    Returns
    -------
    The l2 tolerance corresponding to tol (or the other way around if input_is_l2 == True)

    Notes
    -----
    The finite volume solution SOL is given by the vector sol where sol_i is the constant value assigned to element i of the grid.
    For a uniform grid, the volume vol_i of grid element i is the same for all elements, so vol_i = vol = vol_{\Omega} / {num elements}.
    We thus have
    ||SOL||_{L^2}^2 = \int_{\Omega} {SOL(x)}^2 dx = \sum_{i=1}^{num elements} {sol_i]}^2 vol_i = vol \sum_{i=1}^{num elements} {sol_i}^2 = vol ||sol||_{l^2}^2
    """
    if "SourceBeam" in testcase:
        vol_domain = 3.
        num_elements = grid_size
    elif "Checkerboard3d" in testcase:
        vol_domain = 7.**3
        num_elements = grid_size**3
    else:
        raise NotImplementedError("")
    vol = vol_domain / num_elements
    return tol / math.sqrt(vol) if not input_is_l2 else tol * math.sqrt(vol)


def create_and_scatter_sourcebeam_parameters(comm, min_param, max_param):
    ''' Samples all 3 parameters uniformly with the same width and adds random parameter combinations until
        comm.Get_size() parameters are created. After that, parameter combinations are scattered to ranks. '''
    num_samples_per_parameter = int(comm.Get_size()**(1. / 3.) + 0.1)
    sample_width = (max_param - min_param) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sigma_a_left_range = sigma_s_left_range = sigma_s_right_range = np.arange(min_param, max_param + 1e-13,
                                                                              sample_width)
    sigma_a_right_range = sigma_s_middle_range = [0.]
    parameters_list = []
    for sigma_a_left in sigma_a_left_range:
        for sigma_a_right in sigma_a_right_range:
            for sigma_s_left in sigma_s_left_range:
                for sigma_s_middle in sigma_s_middle_range:
                    for sigma_s_right in sigma_s_right_range:
                        parameters_list.append(
                            [sigma_a_left, sigma_a_right, sigma_s_left, sigma_s_middle, sigma_s_right])
    while len(parameters_list) < comm.Get_size():
        parameters_list.append([
            random.uniform(min_param, max_param), 0.,
            random.uniform(min_param, max_param), 0.,
            random.uniform(min_param, max_param)
        ])
    return comm.scatter(parameters_list, root=0)


def create_and_scatter_parameters(testcase, comm, min_param=1., max_param=8.):
    if "Checkerboard" in testcase:
        return create_and_scatter_boltzmann_parameters(comm, min_param, max_param)
    elif "SourceBeam" in testcase:
        return create_and_scatter_sourcebeam_parameters(comm, min_param, max_param)
    else:
        raise NotImplementedError("Unknown testcase!")


def create_coordinatetransformedmn_solver(gridsize, mu, testcase):
    prefix = "coordinatetransformedmn_sigma_s_s_{}_a_{}_sigma_t_s_{}_a_{}".format(*mu)
    return wrapper.Solver(testcase, prefix, 2000000, gridsize, False, False, mu)


def calculate_trajectory_l2_error(final_modes, grid_size, mu, testcase):
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
    _, snapshots_alpha = solver.solve()
    # compute errors in alpha coordinates
    # snapshots_alpha_np = NumpyVectorSpace(snapshots_alpha[0].dim).from_data(snapshots_alpha.data)
    # differences_alpha = snapshots_alpha_np - final_modes.lincomb(snapshots_alpha_np.dot(final_modes))
    # abs_error_alpha  = np.sum(differences_alpha.l2_norm2())
    # rel_error_alpha  = np.sum(differences_alpha.l2_norm()/snapshots_alpha_np.l2_norm())
    projected_snapshots_alpha = final_modes.lincomb(snapshots_alpha.dot(final_modes))
    differences_alpha = snapshots_alpha - projected_snapshots_alpha
    abs_error_alpha = np.sum(differences_alpha.l2_norm2())
    rel_error_alpha = np.sum(differences_alpha.l2_norm() / snapshots_alpha.l2_norm())
    # convert to u coordinates
    snapshots_u = solver.solution_space.make_array([solver.u_from_alpha(vec) for vec in snapshots_alpha._list])
    projected_snapshots_u = solver.solution_space.make_array(
        [solver.u_from_alpha(vec) for vec in projected_snapshots_alpha._list])
    differences_u = snapshots_u - projected_snapshots_u
    abs_error_u = np.sum(differences_u.l2_norm2())
    rel_error_u = np.sum(differences_u.l2_norm() / snapshots_u.l2_norm())
    return abs_error_alpha, rel_error_alpha, abs_error_u, rel_error_u


def calculate_total_l2_projection_error(final_modes, grid_size, mu, testcase, total_num_snapshots, mpi_wrapper):
    abs_error_alpha, rel_error_alpha, abs_error_u, rel_error_u = calculate_trajectory_l2_error(
        final_modes, grid_size, mu, testcase)
    abs_errors_alpha = mpi_wrapper.comm_world.gather(abs_error_alpha, root=0)
    rel_errors_alpha = mpi_wrapper.comm_world.gather(rel_error_alpha, root=0)
    abs_errors_u = mpi_wrapper.comm_world.gather(abs_error_u, root=0)
    rel_errors_u = mpi_wrapper.comm_world.gather(rel_error_u, root=0)
    mean_abs_error_alpha = mean_rel_error_alpha = mean_abs_error_u = mean_rel_error_u = 0
    if mpi_wrapper.rank_world == 0:
        mean_abs_error_alpha = np.sqrt(np.sum(abs_errors_alpha) / total_num_snapshots)
        mean_rel_error_alpha = np.sum(rel_errors_alpha) / total_num_snapshots
        mean_abs_error_u = np.sqrt(np.sum(abs_errors_u) / total_num_snapshots)
        mean_rel_error_u = np.sum(rel_errors_u) / total_num_snapshots
    return mean_abs_error_alpha, mean_rel_error_alpha, mean_abs_error_u, mean_rel_error_u


def calculate_mean_errors(final_modes,
                          grid_size,
                          mu,
                          testcase,
                          total_num_snapshots,
                          mpi_wrapper,
                          logfile=None,
                          selected=False):
    """
    Calculates mean projection errors.
    As we cannot store all snapshots due to memory restrictions, the problem is solved again for error calculation.

    Returns
    -----
    abs_l2_error_alpha: sqrt(1 / total_num_snapshots * sum_s ||s - P(s)||_{l^2}^2)
    rel_l2_error_alpha: 1 / total_num_snapshots * sum_s ||s - P(s)||_{l^2} / ||s||_{l^2})
    abs_L2_error_alpha: Absolute L2 error in alpha coordinates. Due to the uniform FV discretization, this is just abs_l2_error_alpha scaled with a constant factor (as a consequence, the relative L2 error is equal to rel_l2_error_alpha and thus not returned separately)
    abs_l2_error_u: sqrt(1 / total_num_snapshots * sum_s ||u(s) - u(P(s))||_{l^2}^2)
    rel_l2_error_u: 1 / total_num_snapshots * sum_s ||u(s) - u(P(s))||_{l^2} / ||u(s)||_{l^2})
    abs_L2_error_u: Absolute L2 error in u coordinates. See abs_L2_error_alpha.
    """
    start = timer()
    abs_l2_error_alpha, rel_l2_error_alpha, abs_l2_error_u, rel_l2_error_u = calculate_total_l2_projection_error(
        final_modes, grid_size, mu, testcase, total_num_snapshots, mpi_wrapper)
    elapsed = timer() - start
    abs_L2_error_alpha = convert_L2_l2(abs_l2_error_alpha, grid_size, testcase, True)
    abs_L2_error_u = convert_L2_l2(abs_l2_error_u, grid_size, testcase, True)
    if mpi_wrapper.rank_world == 0 and logfile is not None:
        logfile.write(f"Time used for calculating error: {elapsed} s\n")
        selected_string = "for selected snaps" if selected else "for all snaps"
        logfile.write(f"Errors {selected_string}:\n")
        logfile.write(f"l2_abs_a l2_rel_a L2_abs_a l2_abs_u l2_rel_u L2_abs_u\n")
        logfile.write(
            f"{abs_l2_error_alpha:.2e} {rel_l2_error_alpha:.2e} {abs_L2_error_alpha:.2e} {abs_l2_error_u:.2e} {rel_l2_error_u:.2e} {abs_L2_error_u:.2e}\n"
        )
    return abs_l2_error_alpha, rel_l2_error_alpha, abs_L2_error_alpha, abs_l2_error_u, rel_l2_error_u, abs_L2_error_u
