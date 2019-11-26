import sys
import numpy as np
from boltzmann.wrapper import CellModelSolver, CellModelPfieldOperator, CellModelOfieldOperator, CellModelStokesOperator
from mpiwrapper import MPIWrapper

if __name__ == "__main__":
    mpi = MPIWrapper()
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    visualize = True if argc < 7 else bool(sys.argv[6])
    subsampling = True if argc < 8 else bool(sys.argv[7])
    filename = "cellmodel_solve_grid_%dx%d" % (grid_size_x, grid_size_y)
    mu = [5e-13, 1., 1.1]
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    num_cells = solver.num_cells

    t = 0
    save_step_counter = 1

    # initial values
    pfield_vecs = []
    ofield_vecs = []
    for kk in range(num_cells):
        pfield_vecs.append(solver.pfield_vector(kk))
        ofield_vecs.append(solver.ofield_vector(kk))
    stokes_vec = solver.stokes_vector()
    solver.visualize('solve_from_dune', 0, 0., subsampling)

    # pfield_op = CellModelPfieldOperator(solver, 0, dt)
    # ofield_op = CellModelOfieldOperator(solver, 0, dt)
    # stokes_op = CellModelStokesOperator(solver)

    i = 0
    while t < t_end - 1e-14:
        # match saving times and t_end_ exactly
        actual_dt = min(dt, t_end - t)

        # do a timestep
        print("Current time: {}".format(t))
        for kk in range(num_cells):
            solver.prepare_pfield_operator(dt, kk)
            pfield_vecs[kk] = solver.solve_pfield(pfield_vecs[kk], kk)
            solver.set_pfield_vec(kk, pfield_vecs[kk])
            solver.prepare_ofield_operator(dt, kk)
            ofield_vecs[kk] = solver.solve_ofield(ofield_vecs[kk], kk)
            solver.set_ofield_vec(kk, ofield_vecs[kk])

        solver.prepare_stokes_operator()
        stokes_vec = solver.solve_stokes()
        solver.set_stokes_vec(stokes_vec)
        i += 1
        t += actual_dt
        solver.visualize('solve_from_dune', i, t, subsampling)

    mpi.comm_world.Barrier()
