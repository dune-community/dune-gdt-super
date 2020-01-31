from numbers import Number
import sys
import numpy as np
from hapod.cellmodel.wrapper import CellModelSolver, DuneCellModel, CellModel
from hapod.mpi import MPIWrapper

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.algorithms.pod import pod
from pymor.operators.constructions import VectorOperator, ProjectedOperator
from pymor.operators.interface import Operator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.defaults import set_defaults
from pymor.core.logger import set_log_levels

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
            assert all(sb is None or sb in ss for sb, ss in zip(source_bases, operator.source.subspaces))
            source_bases = tuple(None if sb is None else sb.copy() for sb in source_bases)
            self.blocked_source_basis = True
            self.source = BlockVectorSpace([ss if sb is None else NumpyVectorSpace(len(sb))
                                            for ss, sb in zip(operator.source.subspaces, source_bases)])

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


class CellModelReductor(ProjectionBasedReductor):
    def __init__(self, fom, pfield_basis, ofield_basis, stokes_basis, check_orthonormality=None, check_tol=None,
                 least_squares_pfield=False, least_squares_ofield=False, least_squares_stokes=False):
        bases = {'pfield': pfield_basis,
                 'ofield': ofield_basis,
                 'stokes': stokes_basis}
        # products = {'pfield': None,
        #             'ofield': None,
        #             'stokes': None}
        super().__init__(fom, bases, {},
                         check_orthonormality=check_orthonormality, check_tol=check_tol)
        self.__auto_init(locals())

    reduce = ProjectionBasedReductor._reduce  # hack to allow bases which are None

    def project_operators(self):
        fom = self.fom
        pfield_basis, ofield_basis, stokes_basis = self.bases['pfield'], self.bases['ofield'], self.bases['stokes']
        projected_operators = {
            'pfield_op': ProjectedSystemOperator(fom.pfield_op,
                                                 pfield_basis if not self.least_squares_pfield else None,
                                                 [pfield_basis, pfield_basis, ofield_basis, stokes_basis]),
            'ofield_op': ProjectedSystemOperator(fom.ofield_op,
                                                 ofield_basis if not self.least_squares_ofield else None,
                                                 [ofield_basis, pfield_basis, ofield_basis, stokes_basis]),
            'stokes_op': ProjectedSystemOperator(fom.stokes_op,
                                                 stokes_basis if not self.least_squares_stokes else None,
                                                 [stokes_basis, pfield_basis, ofield_basis]),
            'initial_pfield': project(fom.initial_pfield, pfield_basis, None),
            'initial_ofield': project(fom.initial_ofield, ofield_basis, None),
            'initial_stokes': project(fom.initial_stokes, stokes_basis, None)
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        raise NotImplementedError

    def build_rom(self, projected_operators, estimator):
        params = {'stagnation_threshold': 0.99, 'stagnation_window': 10, 'maxiter' : 10000, 'relax': 1, 'rtol': 1e-13, 'atol': 1e-13}
        return self.fom.with_(new_type=CellModel,
                              least_squares_pfield=self.least_squares_pfield, least_squares_ofield=self.least_squares_ofield,
                              least_squares_stokes=self.least_squares_stokes,
                              newton_params_pfield=params, newton_params_ofield=params, newton_params_stokes=params,
                              **projected_operators)

    def reconstruct(self, u):  # , basis='RB'):
        pfield_basis, ofield_basis, stokes_basis = self.bases['pfield'], self.bases['ofield'], self.bases['stokes']
        pfield = pfield_basis.lincomb(u._blocks[0].to_numpy()) if pfield_basis is not None else u._blocks[0]
        ofield = ofield_basis.lincomb(u._blocks[1].to_numpy()) if ofield_basis is not None else u._blocks[1]
        stokes = stokes_basis.lincomb(u._blocks[2].to_numpy()) if stokes_basis is not None else u._blocks[2]
        return self.fom.solution_space.make_array([pfield, ofield, stokes])


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
    m = DuneCellModel(solver, dt, t_end)
    U, stages = m.solve(mu=mu, return_stages=True)
    m.visualize(U, subsampling=True)

    pfield, ofield, stokes = U._blocks
    pfield = pfield.copy()
    pfield.append(stages[0])
    # pfield.axpy(-1, m.initial_pfield.as_vector())
    # pfield_basis, pfield_svals = pod(pfield[1:])
    pfield_basis, pfield_svals = pod(pfield)
    # pfield_basis = m.initial_pfield.as_vector()
    # pfield_basis.append(pfield_pod)
    # gram_schmidt(pfield_basis, copy=False)
    # pfield_basis = gram_schmidt(pfield)

    ofield = ofield.copy()
    # ofield.append(stages[1])
    ofield_basis, ofield_svals = pod(ofield)

    stokes = stokes.copy()
    # ofield.append(stages[1])
    stokes_basis, stokes_svals = pod(stokes)

    # from matplotlib import pyplot as plt
    # plt.semilogy(ofield_svals)
    # plt.show()
    # print(len(pfield_basis))
    pfield_basis = pfield_basis[:10]
    # ofield_basis = ofield_basis[:len(ofield_basis)-5]
    # stokes_basis = stokes_basis[:len(stokes_basis)-5]
    # pfield_basis = pfield_basis[:4]
    # ofield_basis = ofield_basis[:4]
    # stokes_basis = stokes_basis[:4]
    reductor = CellModelReductor(m, pfield_basis, None, None,
                                 least_squares_pfield=False,
                                 least_squares_ofield=False,
                                 least_squares_stokes=False,)
    rom = reductor.reduce()
    u, rom_stages = rom.solve(mu, return_stages=True)
    # m.visualize(rom_stages[0], subsampling=True)
    ROM_P_STAGES = reductor.reconstruct(rom.solution_space.make_array([rom_stages[0],
                                                                       u.space.subspaces[1].zeros(len(rom_stages[0])),
                                                                       u.space.subspaces[2].zeros(len(rom_stages[0]))]))
    U_rom = reductor.reconstruct(u)
    print((U._blocks[0]-U_rom._blocks[0]).norm() / U._blocks[0].norm())
    print((U._blocks[1]-U_rom._blocks[1]).norm() / U._blocks[1].norm())
    print((U._blocks[2]-U_rom._blocks[2]).norm() / U._blocks[2].norm())
    m.visualize(U_rom, prefix='U_rom')
    m.visualize(ROM_P_STAGES, prefix='p_stages')
    # U1 = m.solve(mu={'Pa': 1., 'Be': 0.3, 'Ca': 0.1})
    # U2 = m.solve(mu={'Pa': 1., 'Be': 0.3, 'Ca': 1.0})
    # m.visualize(U1, prefix="py_Be0.3Ca0.1Pa1.0", subsampling=True)
    # m.visualize(U2, prefix="1.0Be0.3Ca1.0", subsampling=True)
