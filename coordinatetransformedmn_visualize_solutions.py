import sys

from mpi4py import MPI

from hapod.coordinatetransformedmn.wrapper import Solver, CoordinatetransformedmnModel, CoordinateTransformedmnOperator
from hapod.coordinatetransformedmn.utility import create_and_scatter_parameters


def visualize_solutions(grid_size, testcase, use_python_solve):
    comm_world = MPI.COMM_WORLD
    parameters = create_and_scatter_parameters(testcase, comm_world, min_param=1, max_param=8)
    # parameters = [1, 0, 0, 2, 10]
    # parameters = [8.0, 0.0, 0.0, 0.0, 0.0] # nicht okay
    # parameters = [8.0, 0.0, 8.0, 0.0, 8.0] # okay, aber langsam
    # parameters = [8.0, 0.0, 8.0, 0.0, 1.0]
    print(parameters)
    if "Checkerboard" in testcase:
        prefix = "{}_sigma_s1_{:g}_s2_{:g}_a1_{:g}_a2_{:g}".format(testcase, *parameters)
    elif "SourceBeam" in testcase:
        prefix = "{}_params_{:g}_{:g}_{:g}_{:g}_{:g}".format(testcase, *parameters)
    else:
        raise NotImplementedError("Unknown testcase!")
    solver = Solver(testcase, prefix, 100000000, grid_size, True, False, parameters)
    # solver.visualize(solver.get_initial_values()._list[0], f"solve_from_{'python' if use_python_solve else 'cpp'}_0")
    if use_python_solve:
        operator = CoordinateTransformedmnOperator(solver)
        model = CoordinatetransformedmnModel(operator, solver.get_initial_values(), solver.t_end)
        times, results = model._solve()
        for i, vec in enumerate(results._list):
            solver.visualize(vec, f"{prefix}_{i}")
    else:
        solver.solve()
    print(parameters, " done ")

if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    testcase = "HFM50SourceBeam" if argc < 3 else sys.argv[2]
    visualize_solutions(grid_size, testcase, use_python_solve=False)
