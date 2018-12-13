import sys

from mpi4py import MPI

from boltzmann import wrapper
from boltzmannutility import create_and_scatter_boltzmann_parameters


def visualize_solutions(grid_size=50):
    comm_world = MPI.COMM_WORLD
    parameters = create_and_scatter_boltzmann_parameters(comm_world)
    solver = wrapper.Solver("boltzmann_Sigma_s1_%g_s2_%g_a1_%g_a2_%g"
                            % (parameters[0], parameters[1], parameters[2], parameters[3]), 1, grid_size, True, False,
                            parameters[0], parameters[1], parameters[2], parameters[3])
    solver.solve()
    print(parameters, " done ")

if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    visualize_solutions(grid_size)
