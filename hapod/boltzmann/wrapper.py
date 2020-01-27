import math
import numpy as np
import random
from timeit import default_timer as timer
import weakref

from pymor.algorithms.projection import project
from pymor.core.interfaces import abstractmethod
from pymor.models.basic import ModelBase
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import (VectorOperator, ConstantOperator, LincombOperator, LinearOperator,
                                           FixedParameterOperator)
from pymor.parameters.base import Parameter, ParameterType, Parametric
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from gdt.vectors import CommonDenseVector
import gdt.boltzmann

from hapod.xt import DuneXtLaListVectorSpace

#import cvxopt
#from cvxopt import matrix as cvxmatrix

IMPL_TYPES = (CommonDenseVector,)

PARAMETER_TYPE = ParameterType({'s': (4,)})


class Solver(Parametric):

    def __init__(self, *args):
        self.impl = gdt.boltzmann.BoltzmannSolver2d(*args)
        #self.impl = gdt.boltzmann.BoltzmannSolver3d(*args)
        self.last_mu = None
        self.solution_space = DuneXtLaListVectorSpace(self.impl.get_initial_values().dim)
        self.build_parameter_type(PARAMETER_TYPE)

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

    def t_end(self):
        return self.impl.t_end()

    def set_current_time(self, time):
        return self.impl.set_current_time(time)

    def set_current_solution(self, vec):
        return self.impl.set_current_solution(vec)

    def time_step_length(self):
        return self.impl.time_step_length()

    def get_initial_values(self):
        return self.solution_space.make_array([self.impl.get_initial_values()])

    def set_rhs_operator_params(self,
                                sigma_s_scattering=1,
                                sigma_s_absorbing=0,
                                sigma_t_scattering=1,
                                sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_rhs_operator_parameters(*mu)

    def set_rhs_timestepper_params(self,
                                   sigma_s_scattering=1,
                                   sigma_s_absorbing=0,
                                   sigma_t_scattering=1,
                                   sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_rhs_timestepper_parameters(*mu)


class BoltzmannModelBase(ModelBase):

    def __init__(self,
                 lf,
                 rhs,
                 initial_data,
                 nt=60,
                 dt=0.056,
                 t_end=3.2,
                 operators=None,
                 products=None,
                 estimator=None,
                 visualizer=None,
                 parameter_space=None,
                 cache_region=None,
                 name=None):
        super().__init__(products=products,
                         estimator=estimator,
                         visualizer=visualizer,
                         cache_region=cache_region,
                         name=name)
        self.build_parameter_type(PARAMETER_TYPE)
        self.__auto_init(locals())
        self.solution_space = self.initial_data.range

    # def project_to_realizable_set(self, vec, cvxopt_P, cvxopt_G, cvxopt_h, dim, space):
    #    cvxopt_q = cvxmatrix(-vec.to_numpy().transpose(), size=(dim,1), tc='d')
    #    sol = cvxopt.solvers.qp(P=cvxopt_P, q=cvxopt_q, G=cvxopt_G, h=cvxopt_h)
    #    if 'optimal' not in sol['status']:
    #        raise NotImplementedError
    #    return NumpyVectorArray(np.array(sol['x']).reshape(1, dim), space)

    # def is_realizable(self, coeffs, basis):
    #    tol = 1e-8
    #    vec = basis.lincomb(coeffs._data)
    #    return np.all(np.greater_equal(vec._data, tol))

    def _solve(self,
               mu=None,
               return_output=False,
               return_half_steps=False,
               cvxopt_P=None,
               cvxopt_G=None,
               cvxopt_h=None,
               basis=None):
        assert not return_output
        U = self.initial_data.as_vector(mu)
        U_half = U.empty()
        U_last = U.copy()
        rhs = self.rhs.assemble(mu)
        final_dt = self.t_end - (self.nt - 1) * self.dt
        assert final_dt >= 0 and final_dt <= self.dt
        for n in range(self.nt):
            dt = self.dt if n != self.nt - 1 else final_dt
            self.logger.info('Time step {}'.format(n))
            param = Parameter({'t': n * self.dt, 'dt': self.dt})
            param['s'] = mu['s']
            V = U_last - self.lf.apply(U_last, param) * dt
            # if cvxopt_P is not None and not self.is_realizable(V, basis):
            #    V = self.project_to_realizable_set(V, cvxopt_P, cvxopt_G, cvxopt_h, V.dim, V.space)
            if return_half_steps:
                U_half.append(V)
            U_last = V + rhs.apply(V, mu=mu) * dt     # explicit Euler for RHS
            # if cvxopt_P is not None and not self.is_realizable(U_last, basis):
            #    U_last = self.project_to_realizable_set(U_last, cvxopt_P, cvxopt_G, cvxopt_h, V.dim, V.space)
            # matrix exponential for RHS
            # mu['dt'] = dt
            # U_last = rhs.apply(V, mu=mu)
            U.append(U_last)
        if return_half_steps:
            return U, U_half
        else:
            return U


class DuneModel(BoltzmannModelBase):

    def __init__(self, nt=60, dt=0.056, *args):
        self.solver = solver = Solver(*args)
        initial_data = VectorOperator(solver.get_initial_values())
        # lf_operator = LFOperator(self.solver)
        # Todo: rename from lf_operator to kinetic_operator
        lf_operator = KineticOperator(self.solver)
        self.non_decomp_rhs_operator = ndrhs = RHSOperator(self.solver)
        param = solver.parse_parameter([0., 0., 0., 0.])
        affine_part = ConstantOperator(ndrhs.apply(initial_data.range.zeros(), mu=param), initial_data.range)
        rhs_operator = affine_part + \
            LincombOperator(
                [LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
                                                       solver.parse_parameter(mu=[0., 0., 0., 0.])) - affine_part),
                 LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
                                                       solver.parse_parameter(mu=[1., 0., 0., 0.])) - affine_part),
                 LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
                                                       solver.parse_parameter(mu=[0., 1., 0., 0.])) - affine_part),
                 LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
                                                       solver.parse_parameter(mu=[1., 0., 1., 0.])) - affine_part),
                 LinearOperator(FixedParameterOperator(RHSOperator(self.solver),
                                                       solver.parse_parameter(mu=[0., 1., 0., 1.])) - affine_part)],
                [ExpressionParameterFunctional('1 - s[0] - s[1]', PARAMETER_TYPE),
                 ExpressionParameterFunctional('s[0] - s[2]', PARAMETER_TYPE),
                 ExpressionParameterFunctional('s[1] - s[3]', PARAMETER_TYPE),
                 ExpressionParameterFunctional('s[2]', PARAMETER_TYPE),
                 ExpressionParameterFunctional('s[3]', PARAMETER_TYPE)]
            )
        param_space = CubicParameterSpace(PARAMETER_TYPE, 0., 10.)
        super().__init__(initial_data=initial_data,
                         lf=lf_operator,
                         rhs=rhs_operator,
                         t_end=solver.t_end(),
                         nt=nt,
                         dt=dt,
                         parameter_space=param_space,
                         name='DuneModel')

    def _solve(self, mu=None, return_output=False, return_half_steps=False):
        assert not return_output
        return (self.with_(new_type=BoltzmannModelBase,
                           rhs=self.non_decomp_rhs_operator).solve(mu=mu, return_half_steps=return_half_steps))


class DuneOperatorBase(OperatorBase):

    def __init__(self, solver):
        self.solver = solver
        self.linear = solver.linear()
        self.source = self.range = solver.solution_space
        self.dt = solver.time_step_length()


class RestrictedDuneOperatorBase(OperatorBase):

    def __init__(self, solver, source_dim, range_dim):
        self.solver = solver
        self.source = NumpyVectorSpace(source_dim)
        self.range = NumpyVectorSpace(range_dim)
        self.dt = solver.time_step_length()


class LFOperator(DuneOperatorBase):

    linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array([
            self.solver.impl.apply_LF_operator(u.impl, mu['t'] if mu is not None and 't' in mu else 0.,
                                               mu['dt'] if mu is not None and 'dt' in mu else self.dt) for u in U._list
        ])


class RestrictedKineticOperator(RestrictedDuneOperatorBase):

    linear = False

    def __init__(self, solver, dofs):
        self.solver = solver
        self.dofs = dofs
        dofs_as_list = [int(i) for i in dofs]
        self.solver.impl.prepare_restricted_operator(dofs_as_list)
        super(RestrictedKineticOperator, self).__init__(solver, self.solver.impl.len_source_dofs(), len(dofs))

    def apply(self, U, mu=None):
        assert U in self.source
        # hack to ensure realizability for hatfunction moment models
        for vec in U._data:
            vec[np.where(vec < 1e-8)] = 1e-8
        U = DuneXtLaListVectorSpace.from_numpy(U.to_numpy())
        ret = [
            DuneXtLaVector(self.solver.impl.apply_restricted_kinetic_operator(u.impl)).to_numpy(True) for u in U._list
        ]
        return self.range.make_array(ret)


class KineticOperator(DuneOperatorBase):

    def apply(self, U, mu=None):
        assert U in self.source
        if not self.linear:
            # hack to ensure realizability for hatfunction moment models
            for vec in U._data:
                vec[np.where(vec < 1e-8)] = 1e-8
        return self.range.make_array([
            self.solver.impl.apply_kinetic_operator(u.impl,
                                                    float(mu['t']) if mu is not None and 't' in mu else 0.,
                                                    float(mu['dt']) if mu is not None and 'dt' in mu else self.dt)
            for u in U._list
        ])

    def restricted(self, dofs):
        return RestrictedKineticOperator(self.solver, dofs), np.array(self.solver.impl.source_dofs())


class GodunovOperator(DuneOperatorBase):

    linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array([self.solver.impl.apply_godunov_operator(u.impl, 0.) for u in U._list])


class RHSOperator(DuneOperatorBase):

    linear = True

    def __init__(self, solver):
        super(RHSOperator, self).__init__(solver)
        self.build_parameter_type(PARAMETER_TYPE)

    def apply(self, U, mu=None):
        assert U in self.source
        # explicit euler for rhs
        self.solver.set_rhs_operator_params(*map(float, mu['s']))
        return self.range.make_array([self.solver.impl.apply_rhs_operator(u.impl, 0.) for u in U._list])
        # matrix exponential for rhs
        # return self.range.make_array([self.solver.impl.apply_rhs_timestepper(u.impl, 0., mu['dt'][0]) for u in U._list])
        # self.solver.set_rhs_timestepper_params(*map(float, mu['s']))

class BoltzmannRBReductor(ProjectionBasedReductor):

    def __init__(self, fom, RB=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, BoltzmannModelBase)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space, (RB.space, fom.solution_space)
        super().__init__(fom, {'RB': RB}, check_orthonormality=check_orthonormality, check_tol=check_tol)

    def project_operators(self):
        fom = self.fom
        RB = self.bases['RB']
        projected_operators = {
            'lf': project(fom.lf, RB, RB),
            'rhs': project(fom.rhs, RB, RB),
            'initial_data': project(fom.initial_data, RB, None),
            'products': {k: project(v, RB, RB) for k, v in fom.products.items()},
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims['RB']
        projected_operators = {
            'lf': project_to_subbasis(rom.lf, dim, dim),
            'rhs': project_to_subbasis(rom.rhs, dim, dim),
            'initial_data': project_to_subbasis(rom.initial_data, dim, None),
            'products': {k: project_to_subbasis(v, dim, dim) for k, v in rom.products.items()},
        }
        return projected_operators

    def build_rom(self, projected_operators, estimator):
        return self.fom.with_(new_type=BoltzmannModelBase, **projected_operators)

