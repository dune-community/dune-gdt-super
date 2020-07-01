import sys
import numpy as np
from pymor.algorithms.ei import deim
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import ProjectionBasedReductor

from hapod.coordinatetransformedmn.utility import (
    create_parameters,
    convert_L2_l2,
    calculate_mean_errors,
    create_coordinatetransformedmn_solver,
)
from hapod.coordinatetransformedmn.wrapper import CoordinateTransformedmnOperator, CoordinatetransformedmnModel


class CoordinatetransformedmnReductor(ProjectionBasedReductor):

    def __init__(self, fom, RB=None, deim_basis=None):
        assert isinstance(fom, CoordinatetransformedmnModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space
        super().__init__(fom, {'RB': RB}, {'RB': None}, check_orthonormality=True)
        self.deim_basis = deim_basis

    def project_operators(self):
        fom = self.fom
        RB = self.bases['RB']
        if self.deim_basis is not None:
            dofs, _, _ = deim(self.deim_basis, pod=False)
            operator = EmpiricalInterpolatedOperator(fom.operator, dofs, self.deim_basis, False)
        else:
            operator = fom.operator

        projected_operators = {
            'operator':          project(operator, RB, RB),
            'initial_data':      project(fom.initial_data, range_basis=RB, source_basis=None),
        }

        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims['RB']
        product = self.products['RB']

        projected_operators = {
            'operator':          project_to_subbasis(rom.operator, dim, dim),
            'initial_data':      project_to_subbasis(rom.initial_data, dim_range=dim, dim_source=None),
        }
        return projected_operators

    def build_rom(self, projected_operators, estimator):
        fom = self.fom
        return CoordinatetransformedmnModel(
            t_end=fom.t_end, initial_dt=fom.initial_dt, atol=fom.atol, rtol=fom.rtol, name=fom.name + '_reduced',
            **projected_operators
        )


def coordinatetransformedmn_pod(mu_count, grid_size, l2_tol, deim_l2_tol, deim_use_nonlinear_snapshots, testcase, logfile=None):

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mus = create_parameters(testcase, mu_count, min_param=min_param, max_param=max_param)

    all_snapshots = None
    model = None

    for mu in mus:
        solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
        operator = CoordinateTransformedmnOperator(solver)
        if model is None:
            model = CoordinatetransformedmnModel(operator, solver.get_initial_values(), solver.t_end,
                                                 solver.initial_dt())

        if all_snapshots is None:
            all_snapshots = solver.solution_space.empty()
            all_nonlinear_snapshots = solver.solution_space.empty()

        # calculate problem trajectory
        times, snapshots, nonlinear_snapshots = solver.solve(store_operator_evaluations=True)
        _, U, _ = model.solve(mu)
        logfile.write(f'Maximum error between solver.solve and model.solve: {np.max(((snapshots - U).norm()))}\n')
        num_snapshots = len(snapshots)
        assert len(times) == num_snapshots

        all_snapshots.append(snapshots, remove_from_other=True)
        all_nonlinear_snapshots.append(nonlinear_snapshots, remove_from_other=True)
        del solver
        print('******', len(all_snapshots))

    basis, svals = pod(all_snapshots, atol=0.0, rtol=0.0, l2_err=l2_tol * np.sqrt(len(all_snapshots)))
    if logfile is not None:
        logfile.write("After the POD, there are " + str(len(basis)) + " modes of " + str(len(all_snapshots)) + " snapshots left!\n")

    deim_snapshots = all_nonlinear_snapshots if deim_use_nonlinear_snapshots else all_snapshots
    deim_basis, deim_svals = pod(deim_snapshots, atol=0.0, rtol=0.0,
                                 l2_err=deim_l2_tol * np.sqrt(len(deim_snapshots)))
    if logfile is not None:
        logfile.write("After the DEIM-POD, there are " + str(len(deim_basis)) + " modes of " + str(len(deim_snapshots)) + " snapshots left!\n")

    return basis, svals, deim_basis, deim_svals, all_snapshots, mus


def create_model(grid_size, testcase):
    min_param = 1
    max_param = 8
    mu = create_parameters(testcase, 1, min_param=min_param, max_param=max_param)[0]
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
    operator = CoordinateTransformedmnOperator(solver)
    model = CoordinatetransformedmnModel(operator, solver.get_initial_values(), solver.t_end,
                                             solver.initial_dt())
    return model

if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-2 if argc < 3 else float(sys.argv[2])
    deim_L2_tol = L2_tol * 0.01
    deim_use_nonlinear_snapshots = False
    with_deim = True
    testcase = "HFM50SourceBeam" if argc < 4 else sys.argv[3]
    filename = f"{testcase}_POD_gridsize_{grid_size}_tol_{L2_tol}.log"
    logfile = open(filename, "a")
    basis, _, deim_basis, _, all_snapshots, mus = coordinatetransformedmn_pod(
        3, grid_size, convert_L2_l2(L2_tol, grid_size, testcase), convert_L2_l2(deim_L2_tol, grid_size, testcase),
        deim_use_nonlinear_snapshots, testcase, logfile=logfile
    )

    fom = create_model(grid_size, testcase)
    reductor = CoordinatetransformedmnReductor(fom, basis, deim_basis=deim_basis if with_deim else None)
    rom = reductor.reduce()
    for mu in mus:
        _, U, _ = fom.solve(mu)
        _, u, _ = rom.solve(mu)
        U_rb = reductor.reconstruct(u)
        print(convert_L2_l2((U[-1]-U_rb[-1]).norm(), grid_size, testcase, input_is_l2=True),
              convert_L2_l2(U[-1].norm(), grid_size, testcase, input_is_l2=True),
              len(U) - len(U_rb))

    err = convert_L2_l2(np.linalg.norm((all_snapshots - basis.lincomb(all_snapshots.dot(basis))).norm()) / np.sqrt(len(all_snapshots)),
                        grid_size, testcase, input_is_l2=True)
    logfile.write(f'Mean L2-err: {err}\n')
    logfile.close()
    logfile = open(filename, "r")
    print("\n\n\nResults:\n")
    print(logfile.read())
    logfile.close()
