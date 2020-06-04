import sys
import numpy as np

from mpi4py import MPI

from hapod.coordinatetransformedmn.wrapper import Solver, CoordinatetransformedmnModel, CoordinateTransformedmnOperator
from hapod.coordinatetransformedmn.utility import create_and_scatter_parameters


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
    times_diff = [abs(t1 - t2) for t1, t2 in zip(times_cpp, times_python)]
    rel_errors_alpha = [
        (alpha1 - alpha2).l2_norm() / max(alpha1.l2_norm(), alpha2.l2_norm()) if max(alpha1.l2_norm(), alpha2.l2_norm()) > 0.0 else 0.0
        for alpha1, alpha2 in zip(results_python._list, results_cpp._list)
    ]
    comm_world.Barrier()
    max_times_diff = comm_world.reduce(max(times_diff), op=MPI.MAX, root=0)
    max_rel_errors_alpha = comm_world.reduce(max(rel_errors_alpha), op=MPI.MAX, root=0)
    if comm_world.rank == 0:
        error_msg = ""
        if not np.isclose(max_times_diff, 0.0):
            error_msg += f"Timesteps differ, max difference is: {max_times_diff}\n"
        if not np.isclose(max_rel_errors_alpha, 0.0):
            error_msg += f"Alpha vectors differ, max relative l2 error is: {max_rel_errors_alpha}\n"
        if error_msg:
            print(f" nope!\n{error_msg}")
        else:
            print(" yes!")


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    testcase = "HFM50SourceBeam" if argc < 3 else sys.argv[2]
    verbose = True if argc < 4 else not (sys.argv[3] == "False" or sys.argv[3] == "0")
    check_solve(grid_size, testcase, verbose=verbose)