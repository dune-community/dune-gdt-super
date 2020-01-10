import sys

from mpi4py import MPI

from hapod.boltzmann import wrapper
from hapod.boltzmann.utility import create_and_scatter_boltzmann_parameters


def visualize_solutions(grid_size=50):
    comm_world = MPI.COMM_WORLD
    parameters = create_and_scatter_boltzmann_parameters(comm_world)
    solver = wrapper.Solver(
        "boltzmann_snapshot_sigma_s1_%g_s2_%g_a1_%g_a2_%g" % (parameters[0], parameters[1], parameters[2], parameters[3]), 1,
        grid_size, True, False, parameters[0], parameters[1], parameters[2], parameters[3])
    solver.solve()
    print(parameters, " done ")


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 20 if argc < 2 else int(sys.argv[1])
    visualize_solutions(grid_size)
