from hapod.xt import DuneXtLaListVectorSpace
import sys
import numpy as np
from pymor.algorithms.ei import deim
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import ProjectionBasedReductor

from hapod.coordinatetransformedmn.utility import (
    create_parameters,
    convert_L2_l2,
    create_coordinatetransformedmn_solver,
)
from hapod.coordinatetransformedmn.wrapper import CoordinateTransformedmnOperator, CoordinatetransformedmnModel


class CoordinatetransformedmnReductor(ProjectionBasedReductor):
    def __init__(self, fom, RB=None, deim_basis=None):
        assert isinstance(fom, CoordinatetransformedmnModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space
        super().__init__(fom, {"RB": RB}, {"RB": None}, check_orthonormality=True)
        self.deim_basis = deim_basis

    def project_operators(self):
        fom = self.fom
        RB = self.bases["RB"]
        if self.deim_basis is not None:
            dofs, _, _ = deim(self.deim_basis, pod=False)
            operator = EmpiricalInterpolatedOperator(fom.operator, dofs, self.deim_basis, False)
        else:
            operator = fom.operator

        projected_operators = {
            "operator": project(operator, RB, RB),
            "initial_data": project(fom.initial_data, range_basis=RB, source_basis=None),
        }

        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims["RB"]
        product = self.products["RB"]

        projected_operators = {
            "operator": project_to_subbasis(rom.operator, dim, dim),
            "initial_data": project_to_subbasis(rom.initial_data, dim_range=dim, dim_source=None),
        }
        return projected_operators

    def build_rom(self, projected_operators, estimator):
        fom = self.fom
        return CoordinatetransformedmnModel(
            t_end=fom.t_end, initial_dt=fom.initial_dt, atol=fom.atol, rtol=fom.rtol, name=fom.name + "_reduced", **projected_operators
        )


def coordinatetransformedmn_pod(
    mu_count,
    grid_size,
    l2_tol,
    deim_l2_tol,
    deim_use_nonlinear_snapshots,
    testcase,
    logfile=None,
    with_intermediate_results=False,
    use_gram_schmidt=False,
):

    # get boltzmann solver to create snapshots
    min_param = 0
    max_param = 8
    mus = create_parameters(testcase, mu_count, min_param=min_param, max_param=max_param)

    all_snapshots = None
    model = None
    all_nonlinear_snapshots = None
    rk_atol = rk_rtol = 1e-3 if "SourceBeam" in testcase or "PlaneSource" in testcase else 1e-2

    for mu in mus:
        solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
        operator = CoordinateTransformedmnOperator(solver)
        if model is None:
            model = CoordinatetransformedmnModel(
                operator, solver.get_initial_values(), solver.t_end, solver.initial_dt(), atol=rk_atol, rtol=rk_rtol,
            )

        if all_snapshots is None:
            all_snapshots = solver.solution_space.empty()
            all_nonlinear_snapshots = solver.solution_space.empty()

        # calculate problem trajectory
        # times, snapshots, nonlinear_snapshots = solver.solve(store_operator_evaluations=True)
        # _, U, _ = model.solve(mu)
        # logfile.write(f'Maximum error between solver.solve and model.solve: {np.max(((snapshots - U).norm()))}\n')
        times, snapshots, _ = model.solve(mu, include_intermediate_alphas=with_intermediate_results)
        assert len(times) == len(snapshots)

        all_snapshots.append(snapshots, remove_from_other=True)
        # all_nonlinear_snapshots.append(nonlinear_snapshots, remove_from_other=True)
        del solver
        print("******", len(all_snapshots))

    if use_gram_schmidt:
        basis = gram_schmidt(all_snapshots)
        return basis, None, basis, None, all_snapshots, mus
    else:

        basis, svals = pod(all_snapshots, atol=0.0, rtol=0.0, l2_err=l2_tol * np.sqrt(len(all_snapshots)))
        if logfile is not None:
            logfile.write("After the POD, there are " + str(len(basis)) + " modes of " + str(len(all_snapshots)) + " snapshots left!\n")

        if deim_l2_tol is not None:
            deim_snapshots = all_nonlinear_snapshots if deim_use_nonlinear_snapshots else all_snapshots
            deim_basis, deim_svals = pod(deim_snapshots, atol=0.0, rtol=0.0, l2_err=deim_l2_tol * np.sqrt(len(deim_snapshots)))
            if logfile is not None:
                logfile.write(
                    "After the DEIM-POD, there are " + str(len(deim_basis)) + " modes of " + str(len(deim_snapshots)) + " snapshots left!\n"
                )

        if deim_l2_tol is not None:
            return basis, svals, deim_basis, deim_svals, all_snapshots, mus
        else:
            return basis, svals, None, None, all_snapshots, mus


def create_model(grid_size, testcase, mu):
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
    operator = CoordinateTransformedmnOperator(solver)
    rk_atol = rk_rtol = 1e-3 if "SourceBeam" in testcase or "PlaneSource" in testcase else 1e-2
    model = CoordinatetransformedmnModel(
        operator,
        solver.get_initial_values(),
        # 0.30,
        solver.t_end,
        solver.initial_dt(),
        atol=rk_atol,
        rtol=rk_rtol,
    )
    return model


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-2 if argc < 3 else float(sys.argv[2])
    deim_L2_tol = L2_tol * 1e-2 if argc < 4 else float(sys.argv[3])
    testcase = "HFM50SourceBeam" if argc < 5 else sys.argv[4]
    use_gram_schmidt = L2_tol <= 0.0
    with_deim = deim_L2_tol > 0.0
    mu_count = 1 if argc < 6 else int(sys.argv[5])
    with_intermediate_results = False if argc < 7 else (False if sys.argv[6] == "False" else True)
    deim_use_nonlinear_snapshots = False
    filename = f"{testcase}_POD_gridsize_{grid_size}_tol_{L2_tol}.log"
    logfile = open(filename, "a")
    basis, _, deim_basis, _, all_snapshots, mus = coordinatetransformedmn_pod(
        mu_count,
        grid_size,
        convert_L2_l2(L2_tol, grid_size, testcase) if not use_gram_schmidt else None,
        convert_L2_l2(deim_L2_tol, grid_size, testcase) if with_deim else None,
        deim_use_nonlinear_snapshots,
        testcase,
        logfile=logfile,
        with_intermediate_results=with_intermediate_results,
        use_gram_schmidt=use_gram_schmidt,
    )

    min_param = 0
    max_param = 8
    mus = create_parameters(testcase, 1, min_param=min_param, max_param=max_param)
    fom = create_model(grid_size, testcase, mus[0])
    reductor = CoordinatetransformedmnReductor(fom, RB=basis, deim_basis=deim_basis if with_deim else None)
    rom = reductor.reduce()
    for mu in mus:
        _, u, _ = rom.solve(mu)
        _, U, _ = fom.solve(mu)
        U_rb = reductor.reconstruct(u)
        coefficients = U.dot(basis)
        U_projected = basis.lincomb(coefficients)

        solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)
        for i in range(len(U)):
            solver.visualize(U._list[i], f"full_{testcase}_{mu}_{i}")
            solver.visualize(U_projected._list[i], f"projected_{testcase}_{mu}_{i}")
        for i in range(len(U_rb)):
            solver.visualize(U_rb._list[i], f"reduced_{testcase}_{L2_tol}_{deim_L2_tol if with_deim else None}_{mu}_{i}")
        print(
            convert_L2_l2((U[-1] - U_rb[-1]).norm(), grid_size, testcase, input_is_l2=True),
            convert_L2_l2((U[-1] - U_projected[-1]).norm(), grid_size, testcase, input_is_l2=True),
            convert_L2_l2(U[-1].norm(), grid_size, testcase, input_is_l2=True),
            len(U) - len(U_rb),
        )

    err = convert_L2_l2(
        np.linalg.norm((all_snapshots - basis.lincomb(all_snapshots.dot(basis))).norm()) / np.sqrt(len(all_snapshots)),
        grid_size,
        testcase,
        input_is_l2=True,
    )
    logfile.write(f"Mean L2 projection error: {err}\n")
    logfile.close()
    logfile = open(filename, "r")
    print("\n\n\nResults:\n")
    print(logfile.read())
    logfile.close()
