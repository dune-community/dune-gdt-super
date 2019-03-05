import numpy as np

from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import (VectorOperator, ConstantOperator, LincombOperator, LinearOperator,
                                           FixedParameterOperator)
from pymor.parameters.base import Parameter, ParameterType, Parametric
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.vectorarrays.list import VectorInterface, ListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray

import libboltzmann
from libboltzmann import CommonDenseVector

#import cvxopt
#from cvxopt import matrix as cvxmatrix

IMPL_TYPES = (CommonDenseVector,)


PARAMETER_TYPE = ParameterType({'s': (4,)})


class Solver(Parametric):

    def __init__(self, *args):
        #self.impl = libboltzmann.BoltzmannSolver2d(*args)
        self.impl = libboltzmann.BoltzmannSolver3d(*args)
        self.last_mu = None
        self.solution_space = DuneXtLaListVectorSpace(self.impl.get_initial_values().dim())
        self.build_parameter_type(PARAMETER_TYPE)

    def solve(self, with_half_steps=True):
        return self.solution_space.make_array(self.impl.solve(with_half_steps))

    def next_n_time_steps(self, n, with_half_steps=True):
        return self.solution_space.make_array(self.impl.next_n_time_steps(n, with_half_steps))

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

    def set_rhs_operator_params(self, sigma_s_scattering=1, sigma_s_absorbing=0, sigma_t_scattering=1,
                                sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_rhs_operator_parameters(*mu)

    def set_rhs_timestepper_params(self, sigma_s_scattering=1, sigma_s_absorbing=0, sigma_t_scattering=1,
                                   sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_rhs_timestepper_parameters(*mu)


class BoltzmannDiscretizationBase(DiscretizationBase):

    special_operators = frozenset({'lf', 'rhs', 'initial_data'})

    def __init__(self, nt=60, dt=0.056, t_end=3.2, initial_data=None, operators=None,
                 products=None, estimator=None, visualizer=None, parameter_space=None,
                 cache_region=None, name=None, lf=None, rhs=None):
        super(BoltzmannDiscretizationBase, self).__init__(
            operators=operators, products=products,
            estimator=estimator, visualizer=visualizer, cache_region=cache_region, name=name,
            lf=lf, rhs=rhs, initial_data=initial_data
        )
        self.nt = nt
        self.dt = dt
        self.t_end = t_end
        self.solution_space = self.initial_data.range
        self.build_parameter_type(PARAMETER_TYPE)
        self.parameter_space = parameter_space

    #def project_to_realizable_set(self, vec, cvxopt_P, cvxopt_G, cvxopt_h, dim, space):
    #    cvxopt_q = cvxmatrix(-vec.to_numpy().transpose(), size=(dim,1), tc='d')
    #    sol = cvxopt.solvers.qp(P=cvxopt_P, q=cvxopt_q, G=cvxopt_G, h=cvxopt_h)
    #    if 'optimal' not in sol['status']:
    #        raise NotImplementedError
    #    return NumpyVectorArray(np.array(sol['x']).reshape(1, dim), space)

    #def is_realizable(self, coeffs, basis):
    #    tol = 1e-8
    #    vec = basis.lincomb(coeffs._data)
    #    return np.all(np.greater_equal(vec._data, tol))

    def _solve(self, mu=None, return_half_steps=False, cvxopt_P=None, cvxopt_G=None, cvxopt_h=None, basis=None):
        U = self.initial_data.as_vector(mu)
        U_half = U.empty()
        U_last = U.copy()
        rhs = self.rhs.assemble(mu)
        final_dt = self.t_end - (self.nt - 1) * self.dt
        assert final_dt >= 0 and final_dt <= self.dt
        for n in range(self.nt):
            dt = self.dt if n != self.nt - 1 else final_dt
            self.logger.info('Time step {}'.format(n))
            # todo: handle all bases (probably in C++)
            param = Parameter({'t' : n*self.dt, 'dt': self.dt})
            param['s'] = mu['s']
            V = U_last - self.lf.apply(U_last, param) * dt
            #if cvxopt_P is not None and not self.is_realizable(V, basis):
            #    V = self.project_to_realizable_set(V, cvxopt_P, cvxopt_G, cvxopt_h, V.dim, V.space)
            if return_half_steps:
                U_half.append(V)
            U_last = V + rhs.apply(V, mu=mu) * dt # explicit Euler for RHS
            #if cvxopt_P is not None and not self.is_realizable(U_last, basis):
            #    U_last = self.project_to_realizable_set(U_last, cvxopt_P, cvxopt_G, cvxopt_h, V.dim, V.space)
            # matrix exponential for RHS
            # mu['dt'] = dt
            # U_last = rhs.apply(V, mu=mu)
            U.append(U_last)
        if return_half_steps:
            return U, U_half
        else:
            return U

    def as_generic_type(self):
        init_args = {k: getattr(self, k) for k in BoltzmannDiscretizationBase._init_arguments}
        operators = dict(self.operators)
        for on in self.special_operators:
            del operators[on]
        init_args['operators'] = operators
        return BoltzmannDiscretizationBase(**init_args)


class DuneDiscretization(BoltzmannDiscretizationBase):

    def __init__(self, nt=60, dt=0.056, *args):
        self.solver = solver = Solver(*args)
        initial_data = VectorOperator(solver.get_initial_values())
        # lf_operator = LFOperator(self.solver)
        lf_operator = KineticOperator(self.solver) # Todo: rename from lf_operator to kinetic_operator
        self.non_decomp_rhs_operator = ndrhs = RHSOperator(self.solver)
        param = solver.parse_parameter([0., 0., 0., 0.])
        affine_part = ConstantOperator(ndrhs.apply(initial_data.range.zeros(),
                                                   mu=param),
                                       initial_data.range)
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
        super(DuneDiscretization, self).__init__(initial_data=initial_data, lf=lf_operator, rhs=rhs_operator,
                                                 t_end=solver.t_end(), nt=nt, dt=dt,
                                                 parameter_space=param_space, name='DuneDiscretization')

    def _solve(self, mu=None, return_half_steps=False):
        return self.as_generic_type().with_(rhs=self.non_decomp_rhs_operator) \
                                     .solve(mu=mu, return_half_steps=return_half_steps)


class DuneOperatorBase(OperatorBase):

    def __init__(self, solver):
        self.solver = solver
        self.source = self.range = solver.solution_space
        self.dt = solver.time_step_length()


class LFOperator(DuneOperatorBase):

    linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(
            [self.solver.impl.apply_LF_operator(u.impl,
                                                mu['t'] if mu is not None and 't' in mu else 0.,
                                                mu['dt'] if mu is not None and 'dt' in mu else self.dt)
             for u in U._list])


class KineticOperator(DuneOperatorBase):

    linear = False

    def apply(self, U, mu=None):
        assert U in self.source
        # hack to ensure realizability for hatfunction moment models
        for vec in U._data:
           vec[np.where(vec < 1e-8)] = 1e-8
        print(mu)
        return self.range.make_array(
            [self.solver.impl.apply_kinetic_operator(u.impl,
                                                float(mu['t']) if mu is not None and 't' in mu else 0.,
                                                float(mu['dt']) if mu is not None and 'dt' in mu else self.dt)
             for u in U._list])


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


class DuneXtLaVector(VectorInterface):

    def __init__(self, impl):
        self.impl = impl

    def to_numpy(self, ensure_copy=False):
        if ensure_copy:
            return np.frombuffer(self.impl.buffer(), dtype=np.double).copy()
        else:
            return np.frombuffer(self.impl.buffer(), dtype=np.double)

    @classmethod
    def make_zeros(cls, subtype):
        impl = subtype[0](subtype[1], 0.)
        return DuneXtLaVector(impl)

    @property
    def dim(self):
        return self.impl.dim()

    @property
    def subtype(self):
        return (type(self.impl), self.impl.dim())

    @property
    def data(self):
        return np.frombuffer(self.impl.buffer(), dtype=np.double)

    def copy(self, deep=False):
        return DuneXtLaVector(self.impl.copy())

    def scal(self, alpha):
        self.impl.scal(alpha)

    def axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def dot(self, other):
        return self.impl.dot(other.impl)

    def l1_norm(self):
        return self.impl.l1_norm()

    def l2_norm(self):
        return self.impl.l2_norm()

    def l2_norm2(self):
        return self.impl.l2_norm()

    def sup_norm(self):
        return self.impl.sup_norm()

    def dofs(self, dof_indices):
        assert 0 <= np.min(dof_indices)
        assert np.max(dof_indices) < self.dim
        return np.array([self.impl[int(i)] for i in dof_indices])

    def amax(self):
        return self.impl.amax()

    def __add__(self, other):
        return DuneXtLaVector(self.impl + other.impl)

    def __iadd__(self, other):
        self.impl += other.impl
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return DuneXtLaVector(self.impl - other.impl)

    def __isub__(self, other):
        self.impl -= other.impl
        return self

    def __mul__(self, other):
        return DuneXtLaVector(self.impl * other)

    def __neg__(self):
        return DuneXtLaVector(-self.impl)

    def __getstate__(self):
        return type(self.impl), self.data

    def __setstate__(self, state):
        self.impl = state[0](len(state[1]), 0.)
        self.data[:] = state[1]


class DuneXtLaListVectorSpace(ListVectorSpace):

    def __init__(self, dim, id_=None):
        self.dim = dim
        self.id = id_

    def __eq__(self, other):
        return type(other) is DuneXtLaListVectorSpace and self.dim == other.dim and self.id == other.id

    @classmethod
    def space_from_vector_obj(cls, vec, id_):
        return cls(len(vec), id_)

    @classmethod
    def space_from_dim(cls, dim, id_):
        return cls(dim, id_)

    def zero_vector(self):
        return DuneXtLaVector(CommonDenseVector(self.dim, 0))

    def make_vector(self, obj):
        return DuneXtLaVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        v = self.zero_vector()
        v.data[:] = data.copy() if ensure_copy else data
        return v
