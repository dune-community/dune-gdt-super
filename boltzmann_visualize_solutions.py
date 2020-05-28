import sys

from mpi4py import MPI

from hapod.boltzmann import wrapper
from hapod.boltzmann.utility import create_and_scatter_boltzmann_parameters


def visualize_solutions(grid_size=50, dimension=2):
    comm_world = MPI.COMM_WORLD
    parameters = create_and_scatter_boltzmann_parameters(comm_world)
    prefix = "boltzmann_snapshot_sigma_s1_{:g}_s2_{:g}_a1_{:g}_a2_{:g}".format(*parameters)
    solver = wrapper.Solver(dimension, prefix, 1, grid_size, True, False, *parameters, False)
    solver.solve()
    print(parameters, " done ")


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 21 if argc < 2 else int(sys.argv[1])
    dimension = 2 if argc < 3 else int(sys.argv[2])
    visualize_solutions(grid_size, dimension)
