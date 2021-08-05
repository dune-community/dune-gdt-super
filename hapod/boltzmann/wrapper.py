from dune.xt.la import CommonVector
import numpy as np
from pymor.algorithms.projection import project
from pymor.models.interface import Model
from pymor.operators.constructions import (
    ConstantOperator,
    FixedParameterOperator,
    LincombOperator,
    LinearOperator,
    VectorOperator,
)
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu, Parameters, ParametricObject
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace

import gdt.boltzmann
from hapod.xt import DuneXtLaListVectorSpace

NUM_PARAMETERS = 4
PARAMETER_TYPE = Parameters({"s": NUM_PARAMETERS})

TESTCASES_1D = [
    "HFM50SourceBeam",
    "HFM50PlaneSource",
]
TESTCASES_2D = []
TESTCASES_3D = ["HFM66Checkerboard3d"]
AVAILABLE_TESTCASES = TESTCASES_1D + TESTCASES_2D + TESTCASES_3D


class Solver(ParametricObject):
    def __init__(self, testcase, *args):
        print(*args)
        self.testcase = testcase
        if testcase == "HFM50SourceBeam":
            self.impl = gdt.boltzmann.HFM50SourceBeamSolver(*args)
        elif testcase == "HFM50PlaneSource":
            self.impl = gdt.boltzmann.HFM50PlaneSourceSolver(*args)
        elif testcase == "HFM66Checkerboard3d":
            self.impl = gdt.boltzmann.HFM66CheckerboardSolver3d(*args)
        else:
            raise NotImplementedError(
                f"Unknown testcase {testcase}, available testcases:\n{AVAILABLE_TESTCASES}"
            )
        self._last_mu = None
        self.solution_space = DuneXtLaListVectorSpace(self.impl.get_initial_values().dim)
        self.t_end = self.impl.t_end()
        self.dt = self.impl.time_step_length()
        self.parameters_own = PARAMETER_TYPE

    def linear(self):
        return self.impl.linear()

    def solve(self, with_half_steps=True):
        return self.solution_space.make_array(self.impl.solve(with_half_steps))

    def next_n_timesteps(self, n, with_half_steps=True):
        return self.solution_space.make_array(self.impl.next_n_timesteps(n, with_half_steps))

    def reset(self):
        self.impl.reset()

    def finished(self):
        return self.impl.finished()

    def current_time(self):
        return self.impl.current_time()

    def set_current_time(self, time):
        return self.impl.set_current_time(time)

    def set_current_solution(self, vec):
        return self.impl.set_current_solution(vec)

    def get_initial_values(self):
        return self.solution_space.make_array([self.impl.get_initial_values()])

    def set_parameters(self, mu):
        if type(mu) == np.ndarray:
            mu = mu.tolist()
        if mu != self._last_mu:
            self._last_mu = mu
            self.impl.set_parameters(mu)

    def visualize(self, vec, prefix):
        self.impl.visualize(vec.impl, prefix)


class BoltzmannModel(Model):
    def __init__(
        self,
        op,
        rhs,
        initial_data,
        nt,
        dt,
        t_end=3.2,
        operators=None,
        products=None,
        estimator=None,
        visualizer=None,
        cache_region=None,
        name=None,
    ):
        super().__init__(
            products=products,
            estimator=estimator,
            visualizer=visualizer,
            cache_region=cache_region,
            name=name,
        )
        self.__auto_init(locals())
        self.parameters_own = PARAMETER_TYPE
        self.solution_space = self.initial_data.range

    def _compute_solution(self, mu=None, **kwargs):
        return_half_steps = kwargs["return_half_steps"] if "return_half_steps" in kwargs else False
        U = self.initial_data.as_vector(mu)
        U_half = U.empty()
        U_last = U.copy()
        rhs = self.rhs.assemble(mu)
        final_dt = self.t_end - (self.nt - 1) * self.dt
        assert final_dt >= 0 and final_dt <= self.dt
        for n in range(self.nt):
            dt = self.dt if n != self.nt - 1 else final_dt
            self.logger.info("Time step {}".format(n))
            param = Mu({"t": n * self.dt, "dt": self.dt, "s": mu["s"]})
            V = U_last - self.op.apply(U_last, param) * dt
            if return_half_steps:
                U_half.append(V)
            U_last = V + rhs.apply(V, mu=mu) * dt  # explicit Euler for RHS
            U.append(U_last)
        if return_half_steps:
            return U, U_half
        else:
            return U


class DuneModel(BoltzmannModel):
    def __init__(self, nt, dt, *args):
        self.solver = solver = Solver(*args)
        initial_data = VectorOperator(solver.get_initial_values())
        kinetic_operator = KineticOperator(self.solver)
        self.non_decomp_rhs_operator = ndrhs = RHSOperator(self.solver)
        # rhs operator is of the form
        # A(mu) u = A_0 u + sum_i mu_i A_i u + b(mu)
        # where b(mu) = b_0 + sum_i mu_i b_i
        # affine_part is b_0
        # The first operator in the LincombOperator is
        # A_0 u
        # the others are
        # A_i u  + b_i
        param = solver.parameters.parse([0.0] * NUM_PARAMETERS)
        affine_part = ConstantOperator(
            ndrhs.apply(initial_data.range.zeros(), mu=param), initial_data.range
        )
        rhs_operator = affine_part + LincombOperator(
            [
                LinearOperator(
                    FixedParameterOperator(
                        RHSOperator(self.solver), solver.parameters.parse(mu=[0.0, 0.0, 0.0, 0.0])
                    )
                    - affine_part
                ),
                LinearOperator(
                    FixedParameterOperator(
                        RHSOperator(self.solver), solver.parameters.parse(mu=[1.0, 0.0, 0.0, 0.0])
                    )
                    - affine_part
                ),
                LinearOperator(
                    FixedParameterOperator(
                        RHSOperator(self.solver), solver.parameters.parse(mu=[0.0, 1.0, 0.0, 0.0])
                    )
                    - affine_part
                ),
                LinearOperator(
                    FixedParameterOperator(
                        RHSOperator(self.solver), solver.parameters.parse(mu=[0.0, 0.0, 1.0, 0.0])
                    )
                    - affine_part
                ),
                LinearOperator(
                    FixedParameterOperator(
                        RHSOperator(self.solver), solver.parameters.parse(mu=[0.0, 0.0, 0.0, 1.0])
                    )
                    - affine_part
                ),
            ],
            #  LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
            #    solver.parameters.parse(mu=[1., 0., 1., 0.])) - affine_part),
            #  LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
            #    solver.parameters.parse(mu=[0., 1., 0., 1.])) - affine_part)],
            # [ExpressionParameterFunctional('1 - s[0] - s[1]', PARAMETER_TYPE),
            #  ExpressionParameterFunctional('s[0] - s[2]', PARAMETER_TYPE),
            #  ExpressionParameterFunctional('s[1] - s[3]', PARAMETER_TYPE),
            #  ExpressionParameterFunctional('s[2]', PARAMETER_TYPE),
            #  ExpressionParameterFunctional('s[3]', PARAMETER_TYPE)]
            [
                ExpressionParameterFunctional("1 - s[0] - s[1] - s[2] - s[3]", PARAMETER_TYPE),
                ExpressionParameterFunctional("s[0]", PARAMETER_TYPE),
                ExpressionParameterFunctional("s[1]", PARAMETER_TYPE),
                ExpressionParameterFunctional("s[2]", PARAMETER_TYPE),
                ExpressionParameterFunctional("s[3]", PARAMETER_TYPE),
            ],
        )
        super().__init__(
            initial_data=initial_data,
            op=kinetic_operator,
            rhs=rhs_operator,
            t_end=solver.t_end,
            nt=nt,
            dt=dt,
            name="DuneModel",
        )

    def _compute_solution(self, mu=None, **kwargs):
        return self.with_(new_type=BoltzmannModel, rhs=self.non_decomp_rhs_operator).compute(
            mu=mu, **kwargs
        )


class DuneOperator(Operator):
    def __init__(self, solver):
        self.solver = solver
        self.linear = solver.linear()
        self.source = self.range = solver.solution_space
        self.dt = solver.dt


class RestrictedDuneOperator(Operator):
    def __init__(self, solver, source_dim, range_dim):
        self.solver = solver
        self.source = NumpyVectorSpace(source_dim)
        self.range = NumpyVectorSpace(range_dim)
        self.dt = solver.dt


class RestrictedKineticOperator(RestrictedDuneOperator):

    linear = False

    def __init__(self, solver, dofs):
        self.solver = solver
        self.dofs = dofs
        dofs_as_list = [int(i) for i in dofs]
        self.solver.impl.prepare_restricted_operator(dofs_as_list)
        super(RestrictedKineticOperator, self).__init__(
            solver, self.solver.impl.len_source_dofs(), len(dofs)
        )

    def apply(self, U, mu=None):
        assert U in self.source
        # hack to ensure realizability for hatfunction moment models
        for vec in U._data:
            vec[np.where(vec < 1e-8)] = 1e-8
        U = DuneXtLaListVectorSpace.from_numpy(U.to_numpy())
        try:
            ret = [
                CommonVector(self.solver.impl.apply_restricted_kinetic_operator(u)).to_numpy(
                    True
                )
                for u in U._list
            ]
        except:
            print(U.to_numpy())
            print(min(U.to_numpy()))
            print(max(U.to_numpy()))
            raise NotImplementedError
        return self.range.make_array(ret)


class KineticOperator(DuneOperator):
    def apply(self, U, mu=None):
        assert U in self.source
        # if not self.linear:
        # hack to ensure realizability for hatfunction moment models
        # for vec in U._data:
        # vec[np.where(vec < 1e-8)] = 1e-8
        return self.range.make_array(
            [
                self.solver.impl.apply_kinetic_operator(
                    u.impl,
                    float(mu["t"]) if mu is not None and "t" in mu else 0.0,
                    float(mu["dt"]) if mu is not None and "dt" in mu else self.dt,
                )
                for u in U._list
            ]
        )

    def restricted(self, dofs):
        return RestrictedKineticOperator(self.solver, dofs), np.array(
            self.solver.impl.source_dofs()
        )


class RHSOperator(DuneOperator):

    linear = True

    def __init__(self, solver):
        super(RHSOperator, self).__init__(solver)

    def apply(self, U, mu=None):
        assert U in self.source
        self.solver.set_parameters(mu["s"])
        return self.range.make_array(
            [self.solver.impl.apply_rhs_operator(u.impl, 0.0) for u in U._list]
        )


class BoltzmannRBReductor(ProjectionBasedReductor):
    def __init__(self, fom, RB=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, BoltzmannModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space, (RB.space, fom.solution_space)
        super().__init__(
            fom, {"RB": RB}, check_orthonormality=check_orthonormality, check_tol=check_tol
        )

    def project_operators(self):
        fom = self.fom
        RB = self.bases["RB"]
        projected_operators = {
            "op": project(fom.op, RB, RB),
            "rhs": project(fom.rhs, RB, RB),
            "initial_data": project(fom.initial_data, RB, None),
            "products": {k: project(v, RB, RB) for k, v in fom.products.items()},
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims["RB"]
        projected_operators = {
            "op": project_to_subbasis(rom.op, dim, dim),
            "rhs": project_to_subbasis(rom.rhs, dim, dim),
            "initial_data": project_to_subbasis(rom.initial_data, dim, None),
            "products": {k: project_to_subbasis(v, dim, dim) for k, v in rom.products.items()},
        }
        return projected_operators

    def build_rom(self, projected_operators, estimator):
        return self.fom.with_(new_type=BoltzmannModel, **projected_operators)
