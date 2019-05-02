import random
import sys
import time
from timeit import default_timer as timer

from mpi4py import MPI
import numpy as np
from pymor.algorithms.ei import deim
from pymor.reductors.basic import GenericRBReductor
from pymor.operators.constructions import Concatenation, VectorArrayOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from boltzmann.wrapper import DuneDiscretization
from boltzmann_binary_tree_hapod import boltzmann_binary_tree_hapod
from boltzmannutility import solver_statistics, create_boltzmann_solver

#from cvxopt import matrix as cvxmatrix


def calculate_l2_error_for_random_samples(basis,
                                          mpi,
                                          solver,
                                          grid_size,
                                          chunk_size,
                                          seed=MPI.COMM_WORLD.Get_rank(),
                                          params_per_rank=1,
                                          with_half_steps=True,
                                          deim_dofs=None,
                                          deim_cb=None,
                                          hyper_reduction='deim'):
    '''Calculates model reduction and projection error for random parameter'''

    random.seed(seed)

    _, num_time_steps = solver_statistics(solver, chunk_size, with_half_steps)
    nt = int(num_time_steps - 1) if not with_half_steps else int((num_time_steps - 1) / 2)
    elapsed_high_dim = elapsed_red = red_errs = proj_errs = 0.

    # realizable projection in reduced space (for hatfunctions)
    #high_dim = basis.to_numpy().shape[1]
    #red_dim = len(basis)
    #print(high_dim, red_dim)
    #cvxopt_P = cvxmatrix(np.eye(red_dim, dtype=float))
    #cvxopt_G = cvxmatrix(-basis.to_numpy(ensure_copy=True).transpose())
    #cvxopt_h = cvxmatrix(-1e-8, (high_dim, 1))

    num_tested_vecs = 0
    for _ in range(params_per_rank):
        # create random parameter
        mu = [random.uniform(0., 8.), random.uniform(0., 8.), 0., random.uniform(0., 8.)]

        # solve without saving solution to measure time
        d = DuneDiscretization(nt, solver.time_step_length(), '', 0, grid_size, False, True, *mu)
        parsed_mu = d.parse_parameter(mu)
        start = timer()
        d.solve(parsed_mu, return_half_steps=False)
        elapsed_high_dim += timer() - start

        # now create Discretization that saves time steps to calculate error
        d = DuneDiscretization(nt, solver.time_step_length(), '', 2000000, grid_size, False, False, *mu)
        d = d.as_generic_type()
        assert hyper_reduction in ('none', 'projection', 'deim')
        if hyper_reduction == 'projection':
            lf = Concatenation([VectorArrayOperator(deim_cb), VectorArrayOperator(deim_cb, adjoint=True), d.lf])
            d = d.with_(lf=lf)
        elif hyper_reduction == 'deim':
            lf = EmpiricalInterpolatedOperator(d.lf, deim_dofs, deim_cb, False)
            d = d.with_(lf=lf)

        print("Starting reduction ", timer() - start)
        start = timer()
        reductor = GenericRBReductor(d.as_generic_type(), basis, basis_is_orthonormal=True)
        rd = reductor.reduce()
        print("Reduction took ", timer() - start)

        # solve reduced problem
        start = timer()
        # U_rb = rd.solve(mu, cvxopt_P=cvxopt_P, cvxopt_G=cvxopt_G, cvxopt_h=cvxopt_h,basis=basis)
        U_rb = rd.solve(parsed_mu, return_half_steps=False)
        elapsed_red += timer() - start

        # reconstruct high-dimensional solution, calculate l^2 error
        step_n = 1
        curr_step = 0
        solver = create_boltzmann_solver(grid_size, mu)
        solver.reset()     # resets some static variables in C++
        while not solver.finished():
            next_U = solver.next_n_time_steps(step_n, False)
            next_U_rb = reductor.reconstruct(U_rb[curr_step:curr_step + len(next_U)])
            red_errs += np.sum((next_U - next_U_rb).l2_norm()**2)
            proj_errs += np.sum((next_U - basis.lincomb(next_U.dot(basis))).l2_norm()**2)
            num_tested_vecs += len(next_U)
            curr_step += len(next_U)

    elapsed_high_dim /= params_per_rank
    elapsed_red /= params_per_rank
    red_errs /= params_per_rank
    proj_errs /= params_per_rank

    # gather measurements on rank 0
    red_errs = mpi.comm_world.gather(red_errs, root=0)
    proj_errs = mpi.comm_world.gather(proj_errs, root=0)
    num_tested_vecs = mpi.comm_world.allreduce(num_tested_vecs, op=MPI.SUM)
    elapsed_red = mpi.comm_world.gather(elapsed_red, root=0)
    elapsed_high_dim = mpi.comm_world.gather(elapsed_high_dim, root=0)

    # calculate mean l^2 errors and execution times
    print("num_tested_vecs", num_tested_vecs)
    mean_red_err = np.sum(np.sqrt(red_errs)) / num_tested_vecs if mpi.rank_world == 0 else 0.
    mean_proj_err = np.sum(np.sqrt(proj_errs)) / num_tested_vecs if mpi.rank_world == 0 else 0.
    elapsed_red_mean = np.sum(elapsed_red) / len(elapsed_red) if mpi.rank_world == 0 else 0.
    elapsed_high_dim_mean = np.sum(elapsed_high_dim) / len(elapsed_high_dim) if mpi.rank_world == 0 else 0.

    return mean_red_err, mean_proj_err, elapsed_red_mean, elapsed_high_dim_mean


if __name__ == "__main__":
    '''Computes HAPOD to get reduced basis and then calculate projection and model reduction error for random samples'''
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    L2_tol = float(sys.argv[3])
    L2_deim_tol = float(sys.argv[4])
    omega = 0.95 if len(sys.argv) <= 5 else float(sys.argv[5])
    timings = False if len(sys.argv) <= 6 else bool(sys.argv[6])
    print(len(sys.argv))
    # We want to prescribe the mean L^2 error, but the HAPOD works with the l^2 error, so rescale tolerance
    tol_scale_factor = (grid_size / 7)**(3 / 2)
    l2_tol = L2_tol * tol_scale_factor
    l2_deim_tol = L2_deim_tol * tol_scale_factor
    filename = "boltzmann_hapod_deim_mor_gridsize_%d_chunksize_%d_tol_%f_deim_tol_%f_omega_%f" % (
        grid_size, chunk_size, L2_tol, L2_deim_tol, omega)
    logfile = open(filename, "a")
    start = timer()
    (basis, eval_basis, _, _, total_num_snaps, total_num_evals, _, mpi, _, _, _, _, solver) = \
            boltzmann_binary_tree_hapod(grid_size, chunk_size, l2_tol, eval_tol=l2_deim_tol, omega=omega, calc_eval_basis=True, logfile=logfile)
    logfile.close()
    logfile = open(filename, "a")
    elapsed_basis_gen = timer() - start
    basis, win = mpi.shared_memory_bcast_modes(basis, returnlistvectorarray=True)
    if mpi.rank_world == 0:
        deim_dofs, deim_cb, _ = deim(eval_basis, len(eval_basis))
        del eval_basis
    else:
        deim_dofs = deim_cb = None
    deim_dofs = mpi.comm_world.bcast(deim_dofs, root=0)
    deim_cb, deim_win = mpi.shared_memory_bcast_modes(deim_cb, returnlistvectorarray=True)

    # the function returns arrays of l^2 errors, each entry representing results for one snapshot
    red_err, proj_err, elapsed_red, elapsed_high_dim = calculate_l2_error_for_random_samples(
        basis, mpi, solver, grid_size, chunk_size, deim_dofs=deim_dofs, deim_cb=deim_cb, hyper_reduction='deim')

    # convert back to L^2 errors
    red_err = red_err / tol_scale_factor
    proj_err = proj_err / tol_scale_factor
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("\n\n\nResults:\n")
        logfile.write('Creating the bases took %g seconds.\n' % elapsed_basis_gen)
        logfile.write('Solving the high-dimensional problem took %g seconds on average.\n' % elapsed_high_dim)
        logfile.write('Solving the reduced problem took %g seconds on average.\n' % elapsed_red)
        logfile.write('The mean L2 reduction error and mean L2 projection error were %g and %g, respectively.\n' %
                      (red_err, proj_err))
        logfile.write(
            'Basis size and collateral basis size were %g and %g, respectively.\n' % (len(basis), len(deim_cb)))
    logfile.close()
