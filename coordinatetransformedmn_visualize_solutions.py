import sys

from mpi4py import MPI

from hapod.coordinatetransformedmn import wrapper
from hapod.boltzmann.utility import create_and_scatter_boltzmann_parameters

def visualize_solutions(grid_size=21):
    comm_world = MPI.COMM_WORLD
    parameters = create_and_scatter_boltzmann_parameters(comm_world)
    prefix = "coordinatetransformedboltzmann_snapshot_sigma_s1_{:g}_s2_{:g}_a1_{:g}_a2_{:g}".format(*parameters)
    solver = wrapper.Solver(prefix, 1, grid_size, True, False, *parameters)
    solver.solve()
    print(parameters, " done ")


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 21 if argc < 2 else int(sys.argv[1])
    visualize_solutions(grid_size)
