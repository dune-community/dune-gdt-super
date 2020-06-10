import sys
import numpy as np

from mpi4py import MPI

from hapod.coordinatetransformedmn.wrapper import (
    Solver,
    CoordinatetransformedmnModel,
    CoordinateTransformedmnOperator,
    RestrictedCoordinateTransformedmnOperator,
)
from hapod.coordinatetransformedmn.utility import create_and_scatter_parameters


def check_solver_results(comm, times1, results1, times2, results2):
    comm.Barrier()
    times_diff = [abs(t1 - t2) for t1, t2 in zip(times1, times2)]
    rel_errors_alpha = [
        (alpha1 - alpha2).l2_norm() / max(alpha1.l2_norm(), alpha2.l2_norm()) if max(alpha1.l2_norm(), alpha2.l2_norm()) > 0.0 else 0.0
        for alpha1, alpha2 in zip(results1._list, results2._list)
    ]
    max_times_diff = comm.reduce(max(times_diff), op=MPI.MAX, root=0)
    max_rel_errors_alpha = comm.reduce(max(rel_errors_alpha), op=MPI.MAX, root=0)
    if comm.rank == 0:
        error_msg = ""
        if not np.isclose(max_times_diff, 0.0):
            error_msg += f"Timesteps differ, max difference is: {max_times_diff}\n"
        if not np.isclose(max_rel_errors_alpha, 0.0):
            error_msg += f"Alpha vectors differ, max relative l2 error is: {max_rel_errors_alpha}\n"
        if error_msg:
            print(f" nope!\n{error_msg}")
        else:
            print(" yes!")


def check_solve(grid_size, testcase, verbose=False):
    comm_world = MPI.COMM_WORLD
    if comm_world.rank == 0:
        print("Checking whether the C++ and Python versions of solve do exactly the same...", end="", flush=True)
    parameters = create_and_scatter_parameters(testcase, comm_world, min_param=1, max_param=8)
    if "Checkerboard" in testcase:
        prefix = "{}_sigma_s1_{:g}_s2_{:g}_a1_{:g}_a2_{:g}".format(testcase, *parameters)
    elif "SourceBeam" in testcase:
        prefix = "{}_params_{:g}_{:g}_{:g}_{:g}_{:g}".format(testcase, *parameters)
    else:
        raise NotImplementedError("Unknown testcase!")
    solver = Solver(testcase, prefix, 1000000000, grid_size, False, not verbose, parameters)
    operator = CoordinateTransformedmnOperator(solver)
    model = CoordinatetransformedmnModel(operator, solver.get_initial_values(), solver.t_end)
    times_python, results_python = model._solve(verbose=verbose)
    times_cpp, results_cpp = solver.solve()
    check_solver_results(comm_world, times_python, results_python, times_cpp, results_cpp)
    if comm_world.rank == 0:
        print("Checking whether next_n_steps works as expected...", end="", flush=True)
    # create new solver, we can also call reset() once we add such a method
    solver = Solver(testcase, prefix, 1000000000, grid_size, False, not verbose, parameters)
    dt = solver.initial_dt()
    times_next_n_steps = []
    results_next_n_steps = solver.solution_space.empty()
    n = 3
    while not solver.finished():
        next_times, next_results, dt = solver.next_n_steps(n, dt)
        times_next_n_steps += next_times
        results_next_n_steps.append(next_results)
    check_solver_results(comm_world, times_next_n_steps, results_next_n_steps, times_cpp, results_cpp)


def check_restricted_op(grid_size, testcase, verbose=False):
    comm_world = MPI.COMM_WORLD
    if comm_world.rank == 0:
        print("Checking whether the restricted operator gives the correct results...", end="", flush=True)
    parameters = create_and_scatter_parameters(testcase, comm_world, min_param=1, max_param=8)
    # create solver
    solver = Solver(testcase, "", 1000000000, grid_size, False, not verbose, parameters)
    # generate random output_dofs
    from random import seed
    from random import randint, uniform

    seed(1)
    num_dofs = solver.solution_space.dim
    num_range_dofs = num_dofs // 1000 if num_dofs >= 1000 else 1
    range_dofs = [randint(0, num_dofs - 1) for _ in range(num_range_dofs)]
    # create operator and restricted operator
    operator = CoordinateTransformedmnOperator(solver)
    restricted_operator, source_dofs = operator.restricted(range_dofs)
    # get test_vector to apply operators to
    # ensure that most entries are nonzero to avoid masking errors
    source = solver.get_initial_values()
    for i in range(num_dofs):
        source._list[0][i] += uniform(0.0, 10.0)
    restricted_source = restricted_operator.source.make_array([source._list[0][int(dof)] for dof in source_dofs])
    # apply operator and restricted operator and ensure that results are equal
    result = operator.apply(source)
    restricted_result = restricted_operator.apply(restricted_source)
    # check results
    error_msg = ""
    for i, dof in enumerate(range_dofs):
        diff = abs(restricted_result._data[0, i] - result._list[0][dof])
        if not np.isclose(diff, 0.0):
            error_msg += f"Rank {comm_world.rank}| Difference in {i}-th dof {dof}: {diff}\n"
    error_msgs = comm_world.gather(error_msg, root=0)
    if comm_world.rank == 0:
        error_msg = ""
        for msg in error_msgs:
            error_msg += msg
        if error_msg:
            print(f" nope!\n{error_msg}")
        else:
            print(" yes!")


def check_set_parameters(grid_size, testcase, verbose=False):
    comm_world = MPI.COMM_WORLD
    if comm_world.rank == 0:
        print("Checking whether the set_parameters method works...", end="", flush=True)
    parameters = create_and_scatter_parameters(testcase, comm_world, min_param=1, max_param=8)
    # create solver with default parameter and use set_parameters
    solver1 = Solver(testcase, "", 1000000000, grid_size, False, not verbose)
    solver1.set_parameters(parameters)
    # create another solver directly with the correct parameters
    solver2 = Solver(testcase, "", 1000000000, grid_size, False, not verbose, parameters)

    # solve and check that output is the same
    times1, results1 = solver1.solve()
    times2, results2 = solver2.solve()
    check_solver_results(comm_world, times1, results1, times2, results2)


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    testcase = "HFM50SourceBeam" if argc < 3 else sys.argv[2]
    verbose = False if argc < 4 else not (sys.argv[3] == "False" or sys.argv[3] == "0")
    check_solve(grid_size, testcase, verbose=verbose)
    check_restricted_op(grid_size, testcase, verbose=verbose)
    check_set_parameters(grid_size, testcase, verbose=verbose)
