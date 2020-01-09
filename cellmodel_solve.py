import sys
import numpy as np
from boltzmann.wrapper import CellModelSolver, CellModelPfieldOperator, CellModelOfieldOperator, CellModelStokesOperator
from mpiwrapper import MPIWrapper

from pymor.operators.constructions import VectorOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.models.basic import ModelBase


class CellModel(ModelBase):

    def __init__(self, solver, dt, t_end):
        self.__auto_init(locals())
        self.linear = False
        self.solution_space = BlockVectorSpace([solver.pfield_solution_space,
                                                solver.ofield_solution_space,
                                                solver.stokes_solution_space])
        self.pfield_op = CellModelPfieldOperator(solver, 0, dt)
        self.ofield_op = CellModelOfieldOperator(solver, 0, dt)
        self.stokes_op = CellModelStokesOperator(solver)
        self.initial_pfield = VectorOperator(self.solver.pfield_solution_space.make_array([solver.pfield_vector(0)]))
        self.initial_ofield = VectorOperator(self.solver.ofield_solution_space.make_array([solver.ofield_vector(0)]))
        self.initial_stokes = VectorOperator(self.solver.stokes_solution_space.make_array([solver.stokes_vector()]))

    def _solve(self, mu=None, return_output=False):
        assert not return_output

        # initial values
        pfield_vec = self.initial_pfield.as_vector()
        ofield_vec = self.initial_ofield.as_vector()
        stokes_vec = self.initial_stokes.as_vector()

        U_all = self.solution_space.make_array([pfield_vec, ofield_vec, stokes_vec])

        i = 0
        t = 0
        while t < self.t_end - 1e-14:
            # match saving times and t_end_ exactly
            actual_dt = min(dt, self.t_end - t)

            # do a timestep
            print("Current time: {}".format(t))
            U = self.pfield_op.source.subspaces[1].make_array([pfield_vec, ofield_vec, stokes_vec])
            pfield_vec = self.pfield_op.fix_component(1, U).apply_inverse(pfield_vec.zeros())
            U = self.ofield_op.source.subspaces[1].make_array([pfield_vec, ofield_vec, stokes_vec])
            ofield_vec = self.ofield_op.fix_component(1, U).apply_inverse(ofield_vec.zeros())
            U = self.stokes_op.source.subspaces[1].make_array([pfield_vec, ofield_vec])
            stokes_vec = self.stokes_op.fix_component(1, U).apply_inverse(stokes_vec.zeros())
            i += 1
            t += actual_dt
            U = self.pfield_op.source.subspaces[1].make_array([pfield_vec, ofield_vec, stokes_vec])
            U_all.append(U)

        return U_all

    def visualize(self, U, subsampling=True):
        assert U in self.solution_space
        for i in range(len(U)):
            self.solver.set_pfield_vec(0, U._blocks[0]._list[i])
            self.solver.set_ofield_vec(0, U._blocks[1]._list[i])
            self.solver.set_stokes_vec(U._blocks[2]._list[i])
            self.solver.visualize('solve_from_dune', i, i, subsampling)


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
    m = CellModel(solver, dt, t_end)
    U = m.solve()
    m.visualize(U)
