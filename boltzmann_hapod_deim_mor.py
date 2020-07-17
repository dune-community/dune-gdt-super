import random
import sys
import time
from timeit import default_timer as timer

from mpi4py import MPI
import numpy as np
from pymor.algorithms.ei import deim
from pymor.operators.constructions import Concatenation, VectorArrayOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from hapod.boltzmann.wrapper import DuneModel, BoltzmannRBReductor, BoltzmannModel
from boltzmann_binary_tree_hapod import boltzmann_binary_tree_hapod
from hapod.boltzmann.utility import solver_statistics, create_boltzmann_solver

from hapod.coordinatetransformedmn.utility import convert_L2_l2


def calculate_errors_for_parameter(
    mu, basis, solver, testcase, grid_size, L2_tol, L2_deim_tol, nt, deim_dofs, deim_cb, hyper_reduction, visualize=True
):

    time_hd = time_red = abs_red_err = rel_red_err = abs_proj_err = rel_proj_err = 0.0
    # solve without saving solution to measure time
    fom = DuneModel(nt, solver.dt, testcase, "", 0, grid_size, False, True, mu, solver.dt)
    parsed_mu = fom.parameters.parse(mu)
    start = timer()
    fom.solve(parsed_mu, return_half_steps=False)
    time_hd = timer() - start

    # now create Model that saves time steps to calculate error
    fom = DuneModel(nt, solver.dt, testcase, "", 2000000, grid_size, False, False, mu, solver.dt)
    assert hyper_reduction in ("none", "projection", "deim")
    op = None
    if hyper_reduction == "projection":
        op = Concatenation([VectorArrayOperator(deim_cb), VectorArrayOperator(deim_cb, adjoint=True), fom.op])
    elif hyper_reduction == "deim":
        op = EmpiricalInterpolatedOperator(fom.op, deim_dofs, deim_cb, False)
    fom = fom.with_(new_type=BoltzmannModel, op=op)

    reductor = BoltzmannRBReductor(fom, basis)
    rom = reductor.reduce()

    # solve reduced problem
    start = timer()
    u_rb = rom.solve(parsed_mu, return_half_steps=False)
    time_red = timer() - start

    # reconstruct high-dimensional solution, calculate l^2 error (and visualize if requested)
    step_n = 1
    curr_step = 0
    solver.reset()  # resets some static variables in C++
    solver.set_parameters(mu)
    while not solver.finished():
        next_U = solver.next_n_timesteps(step_n, False)
        next_U_rb = reductor.reconstruct(u_rb[curr_step : curr_step + len(next_U)])
        differences_red = next_U - next_U_rb
        abs_red_err += np.sum(differences_red.l2_norm2())
        rel_red_err += np.sum((differences_red.l2_norm() / next_U.l2_norm()))
        next_U_proj = basis.lincomb(next_U.dot(basis))
        differences_proj = next_U - next_U_proj
        abs_proj_err += np.sum(differences_proj.l2_norm2())
        rel_proj_err += np.sum(differences_proj.l2_norm() / next_U.l2_norm())
        if visualize:
            for step in range(len(next_U)):
                solver.visualize(next_U._list[step], f"full_{testcase}_{grid_size}_{mu}_{curr_step + step}")
                solver.visualize(next_U_proj._list[step], f"projected_{testcase}_{grid_size}_{mu}_{curr_step + step}")
                solver.visualize(
                    next_U_rb._list[step], f"red_{testcase}_{grid_size}_{L2_tol}_{L2_deim_tol}_{mu}_{curr_step + step}"
                )
        curr_step += len(next_U)
    return time_hd, time_red, abs_red_err, rel_red_err, abs_proj_err, rel_proj_err, curr_step


def calculate_errors(
    basis,
    mpi,
    solver,
    testcase,
    grid_size,
    L2_tol,
    L2_deim_tol,
    chunk_size,
    mu,
    seed=MPI.COMM_WORLD.Get_rank(),
    params_per_rank=1,
    with_half_steps=True,
    deim_dofs=None,
    deim_cb=None,
    hyper_reduction="deim",
):
    """Calculates model reduction and projection error for random parameter"""

    random.seed(seed)

    _, num_time_steps = solver_statistics(solver, chunk_size, with_half_steps)
    nt = int(num_time_steps - 1) if not with_half_steps else int((num_time_steps - 1) / 2)
    time_hd = time_red = 0.0
    # errors[0] = mean absolute reduction error for trained mus
    # errors[1] = relative absolute reduction error for trained mus
    # errors[2] = mean absolute projection error for trained mus
    # errors[3] = relative absolute projection error for trained mus
    # errors[4] = mean absolute reduction error for new mus
    # errors[5] = relative absolute reduction error for new mus
    # errors[6] = mean absolute projection error for new mus
    # errors[7] = relative absolute projection error for new mus
    errors = [0.0] * 8

    # calculate error for trained mu
    (
        time_hd_mu,
        time_red_mu,
        abs_red_err_trained,
        rel_red_err_trained,
        abs_proj_err_trained,
        rel_proj_err_trained,
        num_snapshots_mu,
    ) = calculate_errors_for_parameter(
        mu, basis, solver, testcase, grid_size, L2_tol, L2_deim_tol, nt, deim_dofs, deim_cb, hyper_reduction
    )
    time_hd += time_hd_mu
    time_red += time_red_mu
    abs_red_errs_trained = mpi.comm_world.gather(abs_red_err_trained, root=0)
    rel_red_errs_trained = mpi.comm_world.gather(rel_red_err_trained, root=0)
    abs_proj_errs_trained = mpi.comm_world.gather(abs_proj_err_trained, root=0)
    rel_proj_errs_trained = mpi.comm_world.gather(rel_proj_err_trained, root=0)
    num_snapshots = mpi.comm_world.allreduce(num_snapshots_mu, op=MPI.SUM)
    errors[0] = np.sqrt(np.sum(abs_red_errs_trained) / num_snapshots) if mpi.rank_world == 0 else 0.0
    errors[1] = np.sum(rel_red_errs_trained) / num_snapshots if mpi.rank_world == 0 else 0.0
    errors[2] = np.sqrt(np.sum(abs_proj_errs_trained) / num_snapshots) if mpi.rank_world == 0 else 0.0
    errors[3] = np.sum(rel_proj_errs_trained) / num_snapshots if mpi.rank_world == 0 else 0.0

    # calculate errors for new mus
    num_snapshots = 0
    abs_red_err = rel_red_err = abs_proj_err = rel_proj_err = 0.0
    for _ in range(params_per_rank):
        mu = [random.uniform(0.0, 8.0), random.uniform(0.0, 8.0), 0.0, random.uniform(0.0, 8.0)]
        (
            time_hd_mu,
            time_red_mu,
            abs_red_err_mu,
            rel_red_err_mu,
            abs_proj_err_mu,
            rel_proj_err_mu,
            num_snapshots_mu,
        ) = calculate_errors_for_parameter(
            mu, basis, solver, testcase, grid_size, L2_tol, L2_deim_tol, nt, deim_dofs, deim_cb, hyper_reduction
        )
        num_snapshots += num_snapshots_mu
        time_hd += time_hd_mu
        time_red += time_red_mu
        abs_red_err += abs_red_err_mu
        rel_red_err += rel_red_err_mu
        abs_proj_err += abs_proj_err_mu
        rel_proj_err += rel_proj_err_mu

    abs_red_errs = mpi.comm_world.gather(abs_red_err, root=0)
    rel_red_errs = mpi.comm_world.gather(rel_red_err, root=0)
    abs_proj_errs = mpi.comm_world.gather(abs_proj_err, root=0)
    rel_proj_errs = mpi.comm_world.gather(rel_proj_err, root=0)
    num_snapshots = mpi.comm_world.allreduce(num_snapshots, op=MPI.SUM)
    errors[4] = np.sqrt(np.sum(abs_red_errs) / num_snapshots) if mpi.rank_world == 0 else 0.0
    errors[5] = np.sum(rel_red_errs) / num_snapshots if mpi.rank_world == 0 else 0.0
    errors[6] = np.sqrt(np.sum(abs_proj_errs) / num_snapshots) if mpi.rank_world == 0 else 0.0
    errors[7] = np.sum(rel_proj_errs) / num_snapshots if mpi.rank_world == 0 else 0.0

    time_hd /= params_per_rank + 1
    time_red /= params_per_rank + 1

    return time_hd, time_red, errors


if __name__ == "__main__":
    """Computes HAPOD to get reduced basis and then calculate projection and model reduction error for random samples"""
    argc = len(sys.argv)
    grid_size = 300 if argc < 2 else int(sys.argv[1])
    chunk_size = 10 if argc < 3 else int(sys.argv[2])
    L2_tol = 1e-3 if argc < 4 else float(sys.argv[3])
    L2_deim_tol = 1e-3 if argc < 5 else float(sys.argv[4])
    omega = 0.95 if argc < 6 else float(sys.argv[5])
    testcase = "HFM50PlaneSource" if argc < 7 else sys.argv[6]
    timings = False if argc < 8 else bool(sys.argv[7])
    filename = "boltzmann_hapod_deim_mor_gridsize_%d_chunksize_%d_tol_%f_deim_tol_%f_omega_%f.log" % (
        grid_size,
        chunk_size,
        L2_tol,
        L2_deim_tol,
        omega,
    )
    logfile = open(filename, "a")
    start = timer()
    (
        basis,
        eval_basis,
        _,
        _,
        total_num_snaps,
        total_num_evals,
        mu,
        mpi,
        _,
        _,
        _,
        _,
        solver,
    ) = boltzmann_binary_tree_hapod(
        testcase,
        grid_size,
        chunk_size,
        convert_L2_l2(L2_tol, grid_size, testcase),
        eval_tol=convert_L2_l2(L2_deim_tol, grid_size, testcase),
        omega=omega,
        calc_eval_basis=True,
        logfile=logfile,
        dt=-1,
    )
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
    mean_time_hd, mean_time_red, errors = calculate_errors(
        basis,
        mpi,
        solver,
        testcase,
        grid_size,
        L2_tol,
        L2_deim_tol,
        chunk_size,
        mu,
        deim_dofs=deim_dofs,
        deim_cb=deim_cb,
        hyper_reduction="deim",
    )

    # convert absolute errors back to L^2 errors (relative errors are equal in L^2 and l^2 norm)
    for index in (0, 2, 4, 6):
        errors[index] = convert_L2_l2(errors[index], grid_size, testcase, input_is_l2=True)
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("\n\n\nResults:\n")
        logfile.write("Creating the bases took %g seconds.\n" % elapsed_basis_gen)
        logfile.write("Solving the high-dimensional problem took %g seconds on average.\n" % mean_time_hd)
        logfile.write("Solving the reduced problem took %g seconds on average.\n" % mean_time_red)
        logfile.write("Errors:\n")
        logfile.write("abs_red_trained rel_red_trained abs_proj_trained rel_proj_trained ")
        logfile.write("abs_red_new rel_red_new abs_proj_new rel_proj_new\n")
        logfile.write(("{:.2e} " * len(errors) + "\n").format(*errors))
        logfile.write(
            "Basis size and collateral basis size were %g and %g, respectively.\n" % (len(basis), len(deim_cb))
        )
    logfile.close()
