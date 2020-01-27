from numbers import Number
import sys
import numpy as np
from hapod.cellmodel.wrapper import CellModelSolver, CellModel
from hapod.mpi import MPIWrapper

from pymor.operators.constructions import ProjectedOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.defaults import set_defaults
from pymor.core.logger import set_log_levels

set_defaults({'pymor.algorithms.newton.newton.atol':1e-12, 'pymor.algorithms.newton.newton.rtol':1e-13})
# set_log_levels({'pymor.algorithms.newton': 'WARN'})


class ProjectedSystemOperator(Operator):

    def __init__(self, operator, range_bases, source_bases):
        if range_bases is None:
            self.blocked_range_basis = False
            self.range = operator.range
        elif isinstance(range_bases, VectorArray):
            assert range_bases in operator.range
            range_bases = range_bases.copy()
            self.blocked_range_basis = False
            self.range = NumpyVectorSpace(len(range_bases))
        else:
            assert len(range_bases) == len(operator.range.subspaces)
            assert all(rb in rs for rb, rs in zip(range_bases, operator.range.subspaces))
            range_bases = tuple(rb.copy() for rb in range_bases)
            self.blocked_range_basis = True
            self.range = BlockVectorSpace([NumpyVectorSpace(len(rb)) for rb in range_bases])

        if source_bases is None:
            self.blocked_source_basis = False
            self.source = operator.source
        elif isinstance(source_bases, VectorArray):
            assert source_bases in operator.source
            source_bases = source_bases.copy()
            self.blocked_source_basis = False
            self.source = NumpyVectorSpace(len(source_bases))
        else:
            assert len(source_bases) == len(operator.source.subspaces)
            assert all(sb in ss for sb, ss in zip(source_bases, operator.source.subspaces))
            source_bases = tuple(sb.copy() for sb in source_bases)
            self.blocked_source_basis = True
            self.source = BlockVectorSpace([NumpyVectorSpace(len(sb)) for sb in source_bases])

        self.__auto_init(locals())
        self.build_parameter_type(operator)
        self.linear = operator.linear

    def apply(self, U, mu=None):
        raise NotImplementedError

    def fix_components(self, idx, U):
        if isinstance(idx, Number):
            idx = (idx,)
        if isinstance(U, VectorArray):
            U = (U,)
        assert len(idx) == len(U)
        assert all(len(u) == 1 for u in U)
        if not self.blocked_source_basis:
            raise NotImplementedError
        U = tuple(self.source_bases[i].lincomb(u.to_numpy()) if self.source_bases[i] is not None else u
                  for i, u in zip(idx, U))
        op = self.operator.fix_components(idx, U)
        if self.blocked_range_basis:
            raise NotImplementedError
        remaining_source_bases = [sb for i, sb in enumerate(self.source_bases) if i not in idx]
        if len(remaining_source_bases) != 1:
            raise NotImplementedError
        return ProjectedOperator(op, self.range_bases, remaining_source_bases[0])


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
    filename = "cellmodel_solve_grid_%dx%d.log" % (grid_size_x, grid_size_y)
    mu = {'Pa': 1., 'Be': 0.3, 'Ca': 0.1}
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    num_cells = solver.num_cells
    m = CellModel(solver, dt, t_end)
    U = m.solve(mu={'Pa': 1, 'Be': 0.3, 'Ca': 0.1})
    m.visualize(U, subsampling=True)
    # U1 = m.solve(mu={'Pa': 1., 'Be': 0.3, 'Ca': 0.1})
    # U2 = m.solve(mu={'Pa': 1., 'Be': 0.3, 'Ca': 1.0})
    # m.visualize(U1, prefix="py_Be0.3Ca0.1Pa1.0", subsampling=True)
    # m.visualize(U2, prefix="1.0Be0.3Ca1.0", subsampling=True)
