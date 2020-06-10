import math
import numpy as np
import random
from timeit import default_timer as timer
import weakref

from pymor.algorithms.projection import project
from pymor.core.base import abstractmethod
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator, ConstantOperator, LincombOperator, LinearOperator, FixedParameterOperator
from pymor.parameters.base import Mu, Parameters, ParametricObject
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from gdt.vectors import CommonDenseVector
import gdt.coordinatetransformedmn

from hapod.xt import DuneXtLaVector, DuneXtLaListVectorSpace

from hapod.boltzmann.wrapper import DuneOperator, RestrictedDuneOperator

# import cvxopt
# from cvxopt import matrix as cvxmatrix

IMPL_TYPES = (CommonDenseVector,)

# Parameters are:
# - Sourcebeam test: (sigma_a_left, sigma_a_right, sigma_s_left, sigma_s_middle, sigma_s_right)
# - Checkerboard test: (sigma_s_scattering, sigma_s_absorbing, sigma_a_scattering, sigma_a_absorbing)
# PARAMETER_TYPE = Parameters({'s': 4})

TESTCASES_1D = ["HFM50SourceBeam", "PMM50SourceBeam", "M50SourceBeam"]
TESTCASES_2D = []
TESTCASES_3D = ["HFM66Checkerboard3d", "PMM128Checkerboard3d", "M3Checkerboard3d"]
AVAILABLE_TESTCASES = TESTCASES_1D + TESTCASES_2D + TESTCASES_3D


class Solver(ParametricObject):
    def __init__(self, testcase, *args):
        if testcase == "HFM50SourceBeam":
            self.impl = gdt.coordinatetransformedmn.HFM50SourceBeamSolver(*args)
        elif testcase == "PMM50SourceBeam":
            self.impl = gdt.coordinatetransformedmn.PMM50SourceBeamSolver(*args)
        elif testcase == "M50SourceBeam":
            self.impl = gdt.coordinatetransformedmn.M50SourceBeamSolver(*args)
        elif testcase == "HFM66Checkerboard3d":
            self.impl = gdt.coordinatetransformedmn.HFM66CheckerboardSolver3d(*args)
        elif testcase == "PMM128Checkerboard3d":
            self.impl = gdt.coordinatetransformedmn.PMM128CheckerboardSolver3d(*args)
        elif testcase == "M3Checkerboard3d":
            self.impl = gdt.coordinatetransformedmn.M3CheckerboardSolver3d(*args)
        else:
            raise NotImplementedError(f"Unknown testcase {testcase}, available testcases:\n{AVAILABLE_TESTCASES}")
        self._last_mu = None
        self.testcase = testcase
        self.dimDomain = 1 if testcase in TESTCASES_1D else 3
        self.dx = self.impl.dx()
        self.solution_space = DuneXtLaListVectorSpace(self.impl.get_initial_values().dim)
        self.parameters_own = {"s": 5} if "SourceBeam" in testcase else {"s": 4}
        self.t_end = self.impl.t_end()
        self.dt = None  # just to reuse some code in, e.g., DuneModel, should be unused

    def current_time(self):
        return self.impl.current_time()

    def finished(self):
        return self.impl.finished()

    def get_initial_values(self):
        return self.solution_space.make_array([self.impl.get_initial_values()])

    def initial_dt(self):
        return self.impl.initial_dt()

    def linear(self):
        return self.impl.linear()

    def next_n_steps(self, n, initial_dt, store_operator_evaluations = False):
        times, snapshots, nonlinear_snapshots, next_dt = self.impl.next_n_steps(n, initial_dt, store_operator_evaluations)
        return times, self.solution_space.make_array(snapshots), self.solution_space.make_array(nonlinear_snapshots), next_dt

    def num_timesteps(self):
        return self.impl.num_timesteps()

    def set_current_solution(self, vec):
        return self.impl.set_current_solution(vec)

    def set_current_time(self, time):
        return self.impl.set_current_time(time)

    def set_parameters(self, mu):
        if mu != self._last_mu:
            self._last_mu = mu
            self.impl.set_parameters(mu)

    def solve(self, store_operator_evaluations = False, do_not_save=False):
        times, snapshots, nonlinear_snapshots = self.impl.solve(store_operator_evaluations, do_not_save)
        return times, self.solution_space.make_array(snapshots), self.solution_space.make_array(nonlinear_snapshots)


    def u_from_alpha(self, alpha_vec):
        return DuneXtLaVector(self.impl.u_from_alpha(alpha_vec.impl))

    def visualize(self, vec, prefix):
        self.impl.visualize(vec.impl, prefix)


class CoordinatetransformedmnModel(Model):
    def __init__(self, operator, initial_data, t_end, operators=None, products=None, estimator=None, visualizer=None, cache_region=None, name=None):
        super().__init__(products=products, estimator=estimator, visualizer=visualizer, cache_region=cache_region, name=name)
        self.__auto_init(locals())
        # self.solution_space = self.initial_data.range
        # Bogacki-Shampine parameters
        self.rk_A = [[0.0, 0.0, 0.0, 0.0], [1 / 2, 0.0, 0.0, 0.0], [0.0, 3 / 4, 0.0, 0.0], [2 / 9, 1 / 3, 4 / 9, 0.0]]
        self.rk_b1 = [2 / 9, 1 / 3, 4 / 9, 0.0]
        self.rk_b2 = [7 / 24, 1 / 4, 1 / 3, 1 / 8]
        self.rk_c = None  # not explicitly time-dependent atm
        self.rk_q = 2

    def _solve(self, mu=None, return_output=False, verbose=False):
        """
        Solve with adaptive Runge-Kutta-Scheme (Bogacki-Shampine).

        Copied and adapted from C++ (dune/gdt/tools/timestepper/adaptive-rungekutta-kinetic.hh)
        """
        assert not return_output
        assert len(self.initial_data) == 1
        Alphas = self.initial_data
        solver = self.operator.solver
        NonlinearSnaps = solver.solution_space.empty()
        # alpha = self.initial_data.as_vector(mu)
        alpha_n = Alphas.copy()
        t = 0
        times = [t]
        dt = solver.initial_dt()
        t_end = solver.t_end
        first_same_as_last = True
        last_stage_of_previous_step = None
        num_stages = len(self.rk_b1)
        atol = rtol = 1e-3 if solver.dimDomain == 1 else 1e-2
        scale_factor_min = 0.2
        scale_factor_max = 5
        stages = []
        for _ in range(num_stages):
            stages.append(alpha_n.copy())
        while t < t_end:
            max_dt = t_end - t
            dt = min(dt, max_dt)
            mixed_error = 1e10
            time_step_scale_factor = 1.0
            first_stage_to_compute = 0
            if first_same_as_last and last_stage_of_previous_step is not None:
                stages[0] = last_stage_of_previous_step
                first_stage_to_compute = 1
            while not mixed_error < 1.0:
                skip_error_computation = False
                dt *= time_step_scale_factor
                for i in range(first_stage_to_compute, num_stages - 1):
                    alpha_tmp = alpha_n.copy()
                    for j in range(0, i):
                        alpha_tmp.axpy(dt * self.rk_A[i][j], stages[j])
                    stages[i] = self.operator.apply(alpha_tmp, mu=mu)
                    if stages[i] is None:
                        mixed_error = 1e10
                        skip_error_computation = True
                        time_step_scale_factor = 0.5
                        break

                if not skip_error_computation:
                    # compute alpha^{n+1}
                    alpha_np1 = alpha_n.copy()
                    for i in range(0, num_stages - 1):
                        alpha_np1.axpy(dt * self.rk_b1[i], stages[i])
                    # calculate last stage
                    stages[num_stages - 1] = self.operator.apply(alpha_np1, mu=mu)
                    if stages[num_stages - 1] is None:
                        mixed_error = 1e10
                        time_step_scale_factor = 0.5
                        continue

                    # calculate second approximations of alpha at timestep n+1.
                    alpha_tmp = alpha_n.copy()
                    for i in range(0, num_stages):
                        alpha_tmp.axpy(dt * self.rk_b2[i], stages[i])

                    nan_found = self.check_for_nan(alpha_tmp, alpha_np1)
                    if nan_found:
                        mixed_error = 1e10
                        time_step_scale_factor = 0.5
                    else:
                        # calculate error
                        # TODO: Most operations should be componentwise, fix
                        mixed_error = 0
                        for a, b in zip(alpha_tmp._list[0], alpha_np1._list[0]):
                            mixed_error = max(mixed_error, abs(a - b) / (atol + max(abs(a), abs(b)) * rtol))
                        # difference = alpha_tmp - alpha_np1
                        # mixed_error = max(abs(alpha_tmp - alpha_np1) / (atol + max(abs(alpha_tmp), abs(alpha_np1)) * rtol))
                        # scale dt to get the estimated optimal time step length
                        time_step_scale_factor = min(max(0.8 * (1.0 / mixed_error) ** (1.0 / (self.rk_q + 1.0)), scale_factor_min), scale_factor_max)
            alpha_n = alpha_np1.copy()
            Alphas.append(alpha_n)
            for nonlinear_snap in stages[1:]:
                NonlinearSnaps.append(nonlinear_snap)
            last_stage_of_previous_step = stages[num_stages - 1]
            if verbose:
                print(f"t={t}, dt={dt}")
            t += dt
            times.append(t)
            dt *= time_step_scale_factor
        return times, Alphas, NonlinearSnaps

    def check_for_nan(self, vec_array1, vec_array2):
        for vec1, vec2 in zip(vec_array1._list, vec_array2._list):
            for val1, val2 in zip(vec1, vec2):
                if math.isnan(val1 + val2):
                    return True
        return False


class RestrictedCoordinateTransformedmnOperator(RestrictedDuneOperator):

    linear = False

    def __init__(self, solver, dofs):
        self.solver = solver
        self.dofs = dofs
        dofs_as_list = [int(i) for i in dofs]
        self.solver.impl.prepare_restricted_operator(dofs_as_list)
        super(RestrictedCoordinateTransformedmnOperator, self).__init__(solver, self.solver.impl.len_source_dofs(), len(dofs))

    def apply(self, Alpha, mu=None):
        assert Alpha in self.source
        Alpha = DuneXtLaListVectorSpace.from_numpy(Alpha.to_numpy())
        ret = [DuneXtLaVector(self.solver.impl.apply_restricted_operator(alpha.impl)).to_numpy(True) for alpha in Alpha._list]
        return self.range.make_array(ret)


class CoordinateTransformedmnOperator(DuneOperator):
    def apply(self, Alpha, mu=None):
        assert Alpha in self.source
        ret = [self.solver.impl.apply_operator(alpha.impl) for alpha in Alpha._list]
        # if an Exception is thrown in C++, apply_operator returns a vector of length 0
        for vec in ret:
            if len(vec) == 0:
                return None
        return self.range.make_array(ret)

    def restricted(self, dofs):
        return RestrictedCoordinateTransformedmnOperator(self.solver, dofs), np.array(self.solver.impl.source_dofs())
