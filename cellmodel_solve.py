from numbers import Number
import sys
import numpy as np
from statistics import mean

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
            self.source = BlockVectorSpace([
                ss if sb is None else NumpyVectorSpace(len(sb))
                for ss, sb in zip(operator.source.subspaces, source_bases)
            ])

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

    def __init__(self,
                 fom,
                 pfield_basis,
                 ofield_basis,
                 stokes_basis,
                 check_orthonormality=None,
                 check_tol=None,
                 least_squares_pfield=False,
                 least_squares_ofield=False,
                 least_squares_stokes=False):
        bases = {'pfield': pfield_basis, 'ofield': ofield_basis, 'stokes': stokes_basis}
        # products = {'pfield': None,
        #             'ofield': None,
        #             'stokes': None}
        super().__init__(fom, bases, {}, check_orthonormality=check_orthonormality, check_tol=check_tol)
        self.__auto_init(locals())

    reduce = ProjectionBasedReductor._reduce     # hack to allow bases which are None

    def project_operators(self):
        fom = self.fom
        pfield_basis, ofield_basis, stokes_basis = self.bases['pfield'], self.bases['ofield'], self.bases['stokes']
        projected_operators = {
            'pfield_op':
            ProjectedSystemOperator(fom.pfield_op, pfield_basis if not self.least_squares_pfield else None,
                                    [pfield_basis, pfield_basis, ofield_basis, stokes_basis]),
            'ofield_op':
            ProjectedSystemOperator(fom.ofield_op, ofield_basis if not self.least_squares_ofield else None,
                                    [ofield_basis, pfield_basis, ofield_basis, stokes_basis]),
            'stokes_op':
            ProjectedSystemOperator(fom.stokes_op, stokes_basis if not self.least_squares_stokes else None,
                                    [stokes_basis, pfield_basis, ofield_basis]),
            'initial_pfield':
            project(fom.initial_pfield, pfield_basis, None),
            'initial_ofield':
            project(fom.initial_ofield, ofield_basis, None),
            'initial_stokes':
            project(fom.initial_stokes, stokes_basis, None)
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        raise NotImplementedError

    def build_rom(self, projected_operators, estimator):
        params = {
            'stagnation_threshold': 0.99,
            'stagnation_window': 3,
            'maxiter': 10000,
            'relax': 1,
            'rtol': 1e-14,
            'atol': 1e-11
        }
        return self.fom.with_(
            new_type=CellModel,
            least_squares_pfield=self.least_squares_pfield,
            least_squares_ofield=self.least_squares_ofield,
            least_squares_stokes=self.least_squares_stokes,
            newton_params_pfield=params,
            newton_params_ofield=params,
            newton_params_stokes=params,
            **projected_operators)

    def reconstruct(self, u):     # , basis='RB'):
        pfield_basis, ofield_basis, stokes_basis = self.bases['pfield'], self.bases['ofield'], self.bases['stokes']
        pfield = pfield_basis.lincomb(u._blocks[0].to_numpy()) if pfield_basis is not None else u._blocks[0]
        ofield = ofield_basis.lincomb(u._blocks[1].to_numpy()) if ofield_basis is not None else u._blocks[1]
        stokes = stokes_basis.lincomb(u._blocks[2].to_numpy()) if stokes_basis is not None else u._blocks[2]
        return self.fom.solution_space.make_array([pfield, ofield, stokes])


if __name__ == "__main__":
    mpi = MPIWrapper()
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 30 if argc < 5 else int(sys.argv[4])
    grid_size_y = 30 if argc < 6 else int(sys.argv[5])
    visualize = True if argc < 7 else bool(sys.argv[6])
    subsampling = False if argc < 8 else bool(sys.argv[7])
    pod_rtol = 1e-13
    include_newton_stages = True
    reduce_pfield = True
    reduce_ofield = True
    reduce_stokes = True

    mus = []
    Ca = 0.1
    Be = 0.3
    for Pa in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.):
        # for Be in (0.3, 3.):
        #for Ca in (0.1, 1.):
        mus.append({'Pa': Pa, 'Be': Be, 'Ca': Ca})
    # mus.append({'Pa': Pa, 'Be': 0.3, 'Ca': 0.1})
    # mu = {'Pa': 1., 'Be': 0.3, 'Ca': 0.1}
    mu = mus[0]
    mu_test_indices = (0, len(mus) // 2, len(mus) - 1)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    num_cells = solver.num_cells
    m = DuneCellModel(solver, dt, t_end)
    U, stages = m.solve(mu=mu, return_stages=True)
    U_full_test = U.copy()
    for i in range(1, len(mus)):
        U_new, stages_new = m.solve(mu=mus[i], return_stages=True)
        for j in range(3):
            U._blocks[j].append(U_new._blocks[j])
            stages[j].append(stages_new[j])
        if i in mu_test_indices:
            for j in range(3):
                U_full_test._blocks[j].append(U_new._blocks[j])
        Be = mus[i]['Be']
        Ca = mus[i]['Ca']
        Pa = mus[i]['Pa']
        m.visualize(U_new, prefix=f"fullorder_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling)
    pfield, ofield, stokes = U._blocks
    if reduce_pfield:
        pfield = pfield.copy()
        if include_newton_stages:
            pfield.append(stages[0])
        full_pfield_basis, pfield_svals = pod(pfield, rtol=pod_rtol)
    # pfield.axpy(-1, m.initial_pfield.as_vector())
    # pfield_basis, pfield_svals = pod(pfield[1:])
    # pfield_basis = m.initial_pfield.as_vector()
    # pfield_basis.append(pfield_pod)
    # gram_schmidt(pfield_basis, copy=False)
    # pfield_basis = gram_schmidt(pfield)

    if reduce_ofield:
        ofield = ofield.copy()
        if include_newton_stages:
            ofield.append(stages[1])
        full_ofield_basis, ofield_svals = pod(ofield, rtol=pod_rtol)
    #ofield_basis = gram_schmidt(ofield)

    if reduce_stokes:
        stokes = stokes.copy()
        if include_newton_stages:
            stokes.append(stages[2])
        full_stokes_basis, stokes_svals = pod(stokes, rtol=pod_rtol)
    # stokes_basis = gram_schmidt(stokes)

    filename = "rom_results_least_squares_grid{}x{}_tend{}_{}_pfield_{}_ofield_{}_stokes_{}.txt".format(
        grid_size_x, grid_size_y, t_end, 'snapsandstages' if include_newton_stages else 'snaps',
        'pod' if reduce_pfield else 'none', 'pod' if reduce_ofield else 'none', 'pod' if reduce_stokes else 'none')

    new_mus = []
    Ca = 0.1
    Be = 0.3
    for Pa in (0.15, 0.55, 0.95):
        new_mus.append({'Pa': Pa, 'Be': Be, 'Ca': Ca})

    with open(filename, 'w') as ff:
        ff.write(f"{filename}\nTrained with {len(mus)} Parameters for Pa: {[param['Pa'] for param in mus]}\n"
                 f"Tested with {len(new_mus)} new Parameters: {[param['Pa'] for param in new_mus]}\n")
        ff.write("num_basis_vecs pfield_trained ofield_trained stokes_trained pfield_new ofield_new stokes_new\n")

    # solve full-order model for new params
    U_new_mus, stages_new_mus = m.solve(mu=new_mus[0], return_stages=True)
    for i in range(1, len(new_mus)):
        U_new, stages_new = m.solve(mu=new_mus[i], return_stages=True)
        for j in range(3):
            U_new_mus._blocks[j].append(U_new._blocks[j])
            stages_new_mus[j].append(stages_new[j])
        Be = new_mus[i]['Be']
        Ca = new_mus[i]['Ca']
        Pa = new_mus[i]['Pa']
        m.visualize(U_new, prefix=f"fullorder_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling)

    basis_size_step = 5
    min_basis_len = basis_size_step
    pfield_basis_max_len = len(full_pfield_basis) if reduce_pfield else np.inf
    ofield_basis_max_len = len(full_ofield_basis) if reduce_ofield else np.inf
    stokes_basis_max_len = len(full_stokes_basis) if reduce_stokes else np.inf
    max_basis_len = min(pfield_basis_max_len, ofield_basis_max_len, stokes_basis_max_len)

    for basis_len in range(min_basis_len, max_basis_len, basis_size_step):

        pfield_basis = full_pfield_basis[:basis_len] if reduce_pfield else None
        ofield_basis = full_ofield_basis[:basis_len] if reduce_ofield else None
        stokes_basis = full_stokes_basis[:basis_len] if reduce_stokes else None

        reduced_prefix = "{}_pfield_{}_ofield_{}_stokes_{}".format(
            'snapsandstages' if include_newton_stages else 'snaps',
            f'pod{len(pfield_basis)}' if reduce_pfield else 'none',
            f'pod{len(ofield_basis)}' if reduce_ofield else 'none',
            f'pod{len(stokes_basis)}' if reduce_stokes else 'none')

        reductor = CellModelReductor(
            m,
            pfield_basis,
            ofield_basis,
            stokes_basis,
            least_squares_pfield=True if reduce_pfield else False,
            least_squares_ofield=True if reduce_ofield else False,
            least_squares_stokes=True if reduce_stokes else False)
        rom = reductor.reduce()

        ################## solve reduced model for trained parameters ####################
        u, rom_stages = rom.solve(mu, return_stages=True)
        #visualize
        u_reconstructed = reductor.reconstruct(u)
        m.visualize(
            u_reconstructed,
            prefix=f"{reduced_prefix}_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}",
            subsampling=subsampling)

        for i in mu_test_indices[1:]:
            u_new, rom_stages_new = rom.solve(mu=mus[i], return_stages=True)
            for j in range(3):
                u._blocks[j].append(u_new._blocks[j])
                rom_stages[j].append(rom_stages_new[j])
            Be = mus[i]['Be']
            Ca = mus[i]['Ca']
            Pa = mus[i]['Pa']
            u_new_reconstructed = reductor.reconstruct(u_new)
            m.visualize(u_new_reconstructed, prefix=f"{reduced_prefix}_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling)
        # ROM_P_STAGES = reductor.reconstruct(rom.solution_space.make_array([rom_stages[0],
        #                                                                    u.space.subspaces[1].zeros(len(rom_stages[0])),
        #                                                                    u.space.subspaces[2].zeros(len(rom_stages[0]))]))
        U_rom = reductor.reconstruct(u)

        ################## test new parameters #######################
        # solve reduced model for new params
        u_new_mus, rom_stages_new_mus = rom.solve(new_mus[0], return_stages=True)
        for i in range(1, len(new_mus)):
            u_new, rom_stages_new = rom.solve(mu=new_mus[i], return_stages=True)
            for j in range(3):
                u_new_mus._blocks[j].append(u_new._blocks[j])
                rom_stages_new_mus[j].append(rom_stages_new[j])
            Be = new_mus[i]['Be']
            Ca = new_mus[i]['Ca']
            Pa = new_mus[i]['Pa']
            u_new_reconstructed = reductor.reconstruct(u_new)
            m.visualize(u_new_reconstructed, prefix=f"{reduced_prefix}_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling)

        U_rom_new_mus = reductor.reconstruct(u_new_mus)

        with open(filename, 'a') as ff:
            pfield_rel_errors = (U_full_test._blocks[0] - U_rom._blocks[0]).norm() / U_full_test._blocks[0].norm()
            ofield_rel_errors = (U_full_test._blocks[1] - U_rom._blocks[1]).norm() / U_full_test._blocks[1].norm()
            stokes_rel_errors = (U_full_test._blocks[2] - U_rom._blocks[2]).norm() / U_full_test._blocks[2].norm()
            pfield_rel_errors_new_mus = (
                U_new_mus._blocks[0] - U_rom_new_mus._blocks[0]).norm() / U_new_mus._blocks[0].norm()
            ofield_rel_errors_new_mus = (
                U_new_mus._blocks[1] - U_rom_new_mus._blocks[1]).norm() / U_new_mus._blocks[1].norm()
            stokes_rel_errors_new_mus = (
                U_new_mus._blocks[2] - U_rom_new_mus._blocks[2]).norm() / U_new_mus._blocks[2].norm()
            ff.write("{} {} {} {} {} {} {}\n".format(
                basis_len, mean([err for err in pfield_rel_errors if not np.isnan(err)]),
                mean([err for err in ofield_rel_errors if not np.isnan(err)]),
                mean([err for err in stokes_rel_errors if not np.isnan(err)]),
                mean([err for err in pfield_rel_errors_new_mus if not np.isnan(err)]),
                mean([err for err in ofield_rel_errors_new_mus if not np.isnan(err)]),
                mean([err for err in stokes_rel_errors_new_mus if not np.isnan(err)])))
