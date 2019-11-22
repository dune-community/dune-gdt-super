import sys
import numpy as np
from boltzmann.wrapper import CellModelSolver, CellModelPfieldOperator, CellModelOfieldOperator, CellModelStokesOperator
from mpiwrapper import MPIWrapper

if __name__ == "__main__":
    mpi = MPIWrapper()
    # testcase = sys.argv[1]
    # t_end = float(sys.argv[2])
    # dt = float(sys.argv[3])
    # grid_size_x = int(sys.argv[4])
    # grid_size_y = int(sys.argv[5])
    # visualize = bool(sys.argv[6]) if len(sys.argv) > 6 else True
    # filename = "cellmodel_solve_grid_%dx%d" % (grid_size_x, grid_size_y)
    # logfile = open(filename, "a")
    testcase = 'single_cell'
    t_end = 0.01
    dt = 0.001
    grid_size_x = 20
    grid_size_y = 5
    mu = [5e-13, 1., 1.1]
    num_cells = 1
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)

    # if (visualize)
    #   solver.visualize(filename, 0, t_, subsampling);
    t = 0
    # next_save_time = min(t_end, t + write_step)
    save_step_counter = 1

    # initial values
    pfield_vec = solver.pfield_vector()
    ofield_vec = solver.ofield_vector()
    stokes_vec = solver.stokes_vector()

    pfield_op = CellModelPfieldOperator(solver, 0, dt)
    ofield_op = CellModelOfieldOperator(solver, 0, dt)
    stokes_op = CellModelStokesOperator(solver)

    while t < t_end - 1e-14:
        # match saving times and t_end_ exactly
        actual_dt = min(dt, t_end - t)

        # do a timestep
        print("Current time: {}".format(t))
        for kk in range(num_cells):
            solver.prepare_pfield_operator(dt, kk)
            pfield_vec = solver.solve_pfield(pfield_vec, kk)
            solver.set_pfield_variables(kk, pfield_vec)
            solver.prepare_ofield_operator(dt, kk)
            ofield_vec = solver.solve_ofield(ofield_vec, kk)
            solver.set_ofield_variables(kk, ofield_vec)

        solver.prepare_stokes_operator()
        stokes_vec = solver.solve_stokes()
        solver.set_stokes_variables(stokes_vec)

        t += actual_dt

    solver.visualize('solve_from_python', 0, t)
    mpi.comm_world.Barrier()