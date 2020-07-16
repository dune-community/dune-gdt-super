import math
import random
from timeit import default_timer as timer

import numpy as np

from hapod.boltzmann.utility import create_boltzmann_parameters
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
        vol_domain = 3.0
        num_elements = grid_size
    elif "PlaneSource" in testcase:
        vol_domain = 2.4
        num_elements = grid_size
    elif "Checkerboard3d" in testcase:
        vol_domain = 7.0 ** 3
        num_elements = grid_size ** 3
    else:
        raise NotImplementedError("")
    vol = vol_domain / num_elements
    return tol / math.sqrt(vol) if not input_is_l2 else tol * math.sqrt(vol)


def create_sourcebeam_parameters(count, min_param, max_param, seed=1):
    """ Samples all 3 parameters uniformly with the same width and adds random parameter combinations until
        count parameters are created. After that, parameter combinations are scattered to ranks. """
    random.seed(seed)
    num_samples_per_parameter = int(count ** (1.0 / 3.0) + 0.1)
    sample_width = (max_param - min_param) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sigma_a_left_range = sigma_s_left_range = sigma_s_right_range = np.arange(min_param, max_param + 1e-13, sample_width)
    sigma_a_right_range = sigma_s_middle_range = [0.0]
    parameters_list = []
    for sigma_a_left in sigma_a_left_range:
        for sigma_a_right in sigma_a_right_range:
            for sigma_s_left in sigma_s_left_range:
                for sigma_s_middle in sigma_s_middle_range:
                    for sigma_s_right in sigma_s_right_range:
                        parameters_list.append([sigma_a_left, sigma_a_right, sigma_s_left, sigma_s_middle, sigma_s_right])
    while len(parameters_list) < count:
        parameters_list.append(
            [random.uniform(min_param, max_param), 0.0, random.uniform(min_param, max_param), 0.0, random.uniform(min_param, max_param)]
        )
    return parameters_list


def create_parameters(testcase, count, min_param=1.0, max_param=8.0, seed=1):
    if "Checkerboard" in testcase or "PlaneSource" in testcase:
        return create_boltzmann_parameters(count, min_param, max_param, seed)
    elif "SourceBeam" in testcase:
        return create_sourcebeam_parameters(count, min_param, max_param, seed)
    else:
        raise NotImplementedError("Unknown testcase!")


def create_and_scatter_parameters(testcase, comm, min_param=1.0, max_param=8.0, seed=1):
    parameters_list = create_parameters(testcase, comm.Get_size(), min_param, max_param, seed)
    return comm.scatter(parameters_list, root=0)


def create_coordinatetransformedmn_solver(gridsize, mu, testcase):
    prefix = "coordinatetransformedmn_sigma_s_s_{}_a_{}_sigma_t_s_{}_a_{}".format(*mu)
    return wrapper.Solver(testcase, prefix, 2000000, gridsize, False, False, mu)


def calculate_trajectory_l2_errors(final_modes, grid_size, mu, testcase, final_eval_modes=None):
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
    n = 10
    dt = solver.initial_dt()
    abs_error_alpha = rel_error_alpha = abs_error_u = rel_error_u = abs_error_alpha_evals = rel_error_alpha_evals = 0
    while not solver.finished():
        _, snapshots_alpha, nonlinear_snapshots_alpha, dt = solver.next_n_steps(n, dt, store_operator_evaluations=(final_eval_modes is not None))
        projected_snapshots_alpha = final_modes.lincomb(snapshots_alpha.dot(final_modes))
        # compute projection error
        differences_alpha = snapshots_alpha - projected_snapshots_alpha
        abs_error_alpha += np.sum(differences_alpha.l2_norm2())
        rel_error_alpha += np.sum(differences_alpha.l2_norm() / snapshots_alpha.l2_norm())
        del differences_alpha
        # convert to u coordinates
        projected_snapshots_u = solver.solution_space.make_array([solver.u_from_alpha(vec) for vec in projected_snapshots_alpha._list])
        del projected_snapshots_alpha
        snapshots_u = solver.solution_space.make_array([solver.u_from_alpha(vec) for vec in snapshots_alpha._list])
        del snapshots_alpha
        differences_u = snapshots_u - projected_snapshots_u
        del projected_snapshots_u
        abs_error_u += np.sum(differences_u.l2_norm2())
        rel_error_u += np.sum(differences_u.l2_norm() / snapshots_u.l2_norm())
        del differences_u
        # calculate projection error for nonlinear snapshots
        if final_eval_modes is not None:
            # TODO: remove the scaling? gives large errors due to the very large entries of nonlinear_snapshots_alpha
            nonlinear_snapshots_alpha.scal(1.0 / nonlinear_snapshots_alpha.l2_norm())
            projected_evals = final_eval_modes.lincomb(nonlinear_snapshots_alpha.dot(final_eval_modes))
            differences_evals = nonlinear_snapshots_alpha - projected_evals
            abs_error_alpha_evals += np.sum(differences_evals.l2_norm2())
            rel_error_alpha_evals += np.sum(differences_evals.l2_norm() / nonlinear_snapshots_alpha.l2_norm())
    # collect errors and return
    ret = [abs_error_alpha, rel_error_alpha, abs_error_u, rel_error_u]
    if final_eval_modes is not None:
        ret += [abs_error_alpha_evals, rel_error_alpha_evals]
    return ret


def calculate_total_l2_projection_error(final_modes, grid_size, mu, testcase, num_snapshots, mpi_wrapper, final_eval_modes=None, num_evals=None):
    # errors = [abs_error_alpha, rel_error_alpha, abs_error_u, rel_error_u, abs_error_alpha_eval, rel_error_alpha_eval]
    errors = calculate_trajectory_l2_errors(final_modes, grid_size, mu, testcase, final_eval_modes=final_eval_modes)
    gathered_errors = mpi_wrapper.comm_world.gather(errors, root=0)
    if mpi_wrapper.rank_world == 0:
        for i in range(len(errors)):
            errors[i] = [err_list[i] for err_list in gathered_errors]
            num_snaps = num_snapshots if i < 4 else num_evals  # the last two errors are for the nonlinear evaluations
            if i in (0, 2, 4):
                # absolute errors
                errors[i] = np.sqrt(np.sum(errors[i]) / num_snaps)
            else:
                # relative errors
                errors[i] = np.sum(errors[i]) / num_snaps
    return errors


def calculate_mean_errors(
    final_modes, grid_size, mu, testcase, total_num_snapshots, mpi_wrapper, final_eval_modes=None, num_evals=None, logfile=None, selected=False
):
    """
    Calculates mean projection errors.
    As we cannot store all snapshots due to memory restrictions, the problem is solved again for error calculation.

    Returns
    -----
    List with length 9, with entries (in this order)
    abs_l2_error_alpha: sqrt(1 / total_num_snapshots * sum_s ||s - P(s)||_{l^2}^2)
    rel_l2_error_alpha: 1 / total_num_snapshots * sum_s ||s - P(s)||_{l^2} / ||s||_{l^2})
    abs_L2_error_alpha: Absolute L2 error in alpha coordinates. Due to the uniform FV discretization, this is just abs_l2_error_alpha scaled with a constant factor (as a consequence, the relative L2 error is equal to rel_l2_error_alpha and thus not returned separately)
    abs_l2_error_u: sqrt(1 / total_num_snapshots * sum_s ||u(s) - u(P(s))||_{l^2}^2)
    rel_l2_error_u: 1 / total_num_snapshots * sum_s ||u(s) - u(P(s))||_{l^2} / ||u(s)||_{l^2})
    abs_L2_error_u: Absolute L2 error in u coordinates. See abs_L2_error_alpha.
    abs_l2_error_alpha_evals: Same as abs_l2_error_alpha, except that is uses the eval modes and eval snapshots. None if final_eval_modes is None.
    rel_l2_error_alpha_evals: See abs_l2_error_alpha_evals.
    abs_L2_error_alpha_evals: See abs_l2_error_alpha_evals.
    """
    start = timer()
    errors = calculate_total_l2_projection_error(
        final_modes, grid_size, mu, testcase, total_num_snapshots, mpi_wrapper, final_eval_modes=final_eval_modes, num_evals=num_evals
    )
    elapsed = timer() - start
    errors.insert(2, convert_L2_l2(errors[0], grid_size, testcase, True))
    errors.insert(5, convert_L2_l2(errors[3], grid_size, testcase, True))
    if final_eval_modes is not None:
        errors.insert(8, convert_L2_l2(errors[6], grid_size, testcase, True))
    if mpi_wrapper.rank_world == 0 and logfile is not None:
        # logfile.write(f"Time used for calculating error: {elapsed} s\n")
        selected_string = "for selected snaps" if selected else "for all snaps"
        logfile.write(f"Errors {selected_string}:\n")
        logfile.write(f"l2_abs_a l2_rel_a L2_abs_a l2_abs_u l2_rel_u L2_abs_u")
        logfile.write("\n" if len(errors) == 6 else " l2_abs_a_eval l2_rel_a_eval L2_abs_a_eval\n")
        logfile.write(("{:.2e} " * len(errors) + "\n").format(*errors))
    return errors
