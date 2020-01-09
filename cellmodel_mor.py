import random
import sys
import time
from timeit import default_timer as timer

from mpi4py import MPI
import numpy as np
from pymor.operators.constructions import Concatenation, VectorArrayOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator

from boltzmann.wrapper import DuneModel, BoltzmannRBReductor
from cellmodel_binary_tree_hapod import cellmodel_binary_tree_hapod
from boltzmannutility import solver_statistics

#from cvxopt import matrix as cvxmatrix

def calculate_l2_error_for_random_samples(basis,
                                          mpi,
                                          solver,
                                          grid_size,
                                          chunk_size,
                                          seed=MPI.COMM_WORLD.Get_rank(),
                                          params_per_rank=2,
                                          eval_basis=None):
    '''Calculates model reduction and projection error for random parameter'''

    random.seed(seed)

    _, num_time_steps = solver_statistics(solver, chunk_size, with_half_steps)
    nt = int(num_time_steps - 1) if not with_half_steps else int((num_time_steps - 1) / 2)
    elapsed_high_dim = elapsed_red = red_errs = proj_errs = 0.

    Re_min = 1e-14
    Re_max = 1e-4
    Fa_min = 0.1
    Fa_max = 10
    xi_min = 0.1
    xi_max = 10
    for _ in range(params_per_rank):

        mu = [random.uniform(Re_min, Re_max), random.uniform(Fa_min, Fa_max), random.uniform(xi_min, xi_max)]

        fom = DuneModel(nt, solver.time_step_length(), '', 2000000, grid_size, False, True, *mu)

        mu = fom.parse_parameter(mu)

        # calculate high-dimensional solution
        start = timer()
        U = fom.solve(mu, return_half_steps=False)
        elapsed_high_dim += timer() - start

        reductor = BoltzmannRBReductor(fom, basis)
        rom = reductor.reduce()

        # solve reduced problem
        start = timer()

        # U_rb = rd.solve(mu, cvxopt_P=cvxopt_P, cvxopt_G=cvxopt_G, cvxopt_h=cvxopt_h,basis=basis)
        U_rb = rom.solve(mu)
        elapsed_red += timer() - start

        # reconstruct high-dimensional solution, calculate error
        U_rb = reductor.reconstruct(U_rb)
        red_errs += np.sum((U - U_rb).l2_norm()**2)
        proj_errs += np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2)

    elapsed_high_dim /= params_per_rank
    elapsed_red /= params_per_rank
    red_errs /= params_per_rank
    proj_errs /= params_per_rank

    red_errs = mpi.comm_world.gather(red_errs, root=0)
    proj_errs = mpi.comm_world.gather(proj_errs, root=0)
    elapsed_red = mpi.comm_world.gather(elapsed_red, root=0)
    elapsed_high_dim = mpi.comm_world.gather(elapsed_high_dim, root=0)
    return red_errs, proj_errs, elapsed_red, elapsed_high_dim


if __name__ == "__main__":
    '''Computes HAPOD to get reduced basis and then calculate projection and model reduction error for random samples'''
    testcase = sys.argv[1]
    t_end = float(sys.argv[2])
    dt = float(sys.argv[3])
    grid_size_x = int(sys.argv[4])
    grid_size_y = int(sys.argv[5])
    tol = float(sys.argv[6])
    chunk_size = int(sys.argv[7])
    omega = float(sys.argv[8])
    inc_gramian = not (sys.argv[9] == "False" or sys.argv[9] == "0") if len(sys.argv) > 9 else True
    filename = "cellmodel_binary_tree_hapod_grid_%dx%d_chunksize_%d_tol_%f_omega_%f" % (grid_size_x, grid_size_y,
                                                                                        chunk_size, tol, omega)
    logfile = open(filename, 'w')
    bases, mu, mpi, solver = cellmodel_binary_tree_hapod(
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
    for r in bases:
      r.wins = [None] * len(r.modes)
      for k in range(len(r.modes)):
        r.modes[k], r.wins[k] = mpi.shared_memory_bcast_modes(r.modes[k], returnlistvectorarray=retlistvecs)

    red_errs, proj_errs, elapsed_red, elapsed_high_dim = calculate_l2_error_for_random_samples(
        bases, mpi, solver, grid_size, chunk_size)

    red_err = np.sqrt(np.sum(red_errs) / total_num_snaps) / grid_size if mpi.rank_world == 0 else None
    proj_err = np.sqrt(np.sum(proj_errs) / total_num_snaps) / grid_size if mpi.rank_world == 0 else None
    elapsed_red_mean = np.sum(elapsed_red) / len(elapsed_red) if mpi.rank_world == 0 else None
    elapsed_high_dim_mean = np.sum(elapsed_high_dim) / len(elapsed_high_dim) if mpi.rank_world == 0 else None
    if mpi.rank_world == 0:
        print("\n\n\nResults:\n")
        print('Solving the high-dimensional problem took %g seconds on average.' % elapsed_high_dim_mean)
        print('Solving the reduced problem took %g seconds on average.' % elapsed_red_mean)
        print('The mean l2 reduction error and mean l2 projection error were %g and %g, respectively.' % (red_err,
                                                                                                          proj_err))

    # free MPI windows
    for r in bases:
        for win in r.wins:
            win.Free()
    mpi.comm_world.Barrier()
    logfile.close()
