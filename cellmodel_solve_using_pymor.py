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
    pfield_vec = solver.pfield_solution_space.make_array(solver.pfield_vector())
    ofield_vec = solver.ofield_solution_space.make_array(solver.ofield_vector())
    stokes_vec = solver.stokes_solution_space.make_array(solver.stokes_vector())

    pfield_op = CellModelPfieldOperator(solver, 0, dt)
    ofield_op = CellModelOfieldOperator(solver, 0, dt)
    stokes_op = CellModelStokesOperator(solver)

    while t < t_end - 1e-14:
        # match saving times and t_end_ exactly
        actual_dt = min(dt, t_end - t)

        # do a timestep
        print("Current time: {}".format(t))
        for k in range(num_cells):
            pfield_vec[k] = pfield_op[k].apply_inverse(pfield_vec[k])
            ofield_op[k].set_other_vecs(pfield_vec=pfield_vec[k], stokes_vec=stokes_vec)
            ofield_vec = ofield_op[k].apply_inverse(ofield_vec)
            solver.set_ofield_variables(k, ofield_vec)

        solver.prepare_stokes_operator()
        stokes_vec = solver.solve_stokes()
        solver.set_stokes_variables(stokes_vec)

        t += actual_dt

        solver.visualize('solve_from_python', 0, t)
    mpi.comm_world.Barrier()

    # logfile.close()
