from numbers import Number
import math
import numpy as np
import random
from timeit import default_timer as timer
import weakref

from pymor.algorithms.newton import newton
from pymor.core.base import abstractmethod
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.operators.constructions import (VectorOperator, ConstantOperator, LincombOperator, LinearOperator,
                                           FixedParameterOperator)
from pymor.parameters.base import Parameter, ParameterType, Parametric
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.list import Vector, ListVectorSpace, ListVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from pymor.vectorarrays.interface import VectorArray

from hapod.xt import DuneXtLaVector, DuneXtLaListVectorSpace

import gdt.cellmodel

# Parameters are Be, Ca, Pa
CELLMODEL_PARAMETER_TYPE = ParameterType({'s': (3,)})


class CellModelSolver(Parametric):

    def __init__(self, testcase, t_end, grid_size_x, grid_size_y, mu):
        self.impl = gdt.cellmodel.CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, False, float(mu['Be']),
                                                float(mu['Ca']), float(mu['Pa']))
        self.last_mu = mu
        self.pfield_solution_space = DuneXtLaListVectorSpace(self.impl.pfield_vec(0).dim)
        self.pfield_numpy_space = NumpyVectorSpace(self.impl.pfield_vec(0).dim)
        self.ofield_solution_space = DuneXtLaListVectorSpace(self.impl.ofield_vec(0).dim)
        self.stokes_solution_space = DuneXtLaListVectorSpace(self.impl.stokes_vec().dim)
        self.build_parameter_type(CELLMODEL_PARAMETER_TYPE)
        self.num_cells = self.impl.num_cells()

    def linear(self):
        return self.impl.linear()

    def solve(self, dt, write=False, write_step=0, filename='', subsampling=True):
        return self.dune_result_to_pymor(self.impl.solve(dt, write, write_step, filename, subsampling))

    def next_n_timesteps(self, n, dt):
        return self.dune_result_to_pymor(self.impl.next_n_timesteps(n, dt))

    def dune_result_to_pymor(self, result):
        ret = []
        nc = self.num_cells
        for k in range(nc):
            ret.append(self.pfield_solution_space.make_array(result[k]))
        for k in range(nc):
            ret.append(self.ofield_solution_space.make_array(result[nc + k]))
        ret.append(self.stokes_solution_space.make_array(result[2 * nc]))
        return ret

    # def reset(self):
    #     self.impl.reset()
    def visualize(self, prefix, num, dt, subsampling=True):
        self.impl.visualize(prefix, num, dt, subsampling)

    def finished(self):
        return self.impl.finished()

    def apply_pfield_product_operator(self, U, mu=None, numpy=False):
        pfield_space = self.pfield_solution_space
        if not numpy:
            U_out = [self.impl.apply_pfield_product_operator(vec.impl) for vec in U._list]
            return pfield_space.make_array(U_out)
        else:
            U_list = pfield_space.make_array([pfield_space.vector_from_numpy(vec).impl for vec in U.to_numpy()])
            U_out = [
                DuneXtLaVector(self.impl.apply_pfield_product_operator(vec.impl)).to_numpy(True)
                for vec in U_list._list
            ]
            return self.pfield_numpy_space.make_array(U_out)

    def apply_ofield_product_operator(self, U, mu=None):
        U_out = [self.impl.apply_ofield_product_operator(vec.impl) for vec in U._list]
        return self.ofield_solution_space.make_array(U_out)

    def apply_stokes_product_operator(self, U, mu=None):
        U_out = [self.impl.apply_stokes_product_operator(vec.impl) for vec in U._list]
        return self.stokes_solution_space.make_array(U_out)

    def pfield_vector(self, cell_index):
        return DuneXtLaVector(self.impl.pfield_vec(cell_index))

    def ofield_vector(self, cell_index):
        return DuneXtLaVector(self.impl.ofield_vec(cell_index))

    def stokes_vector(self):
        return DuneXtLaVector(self.impl.stokes_vec())

    def set_pfield_vec(self, cell_index, vec):
        return self.impl.set_pfield_vec(cell_index, vec.impl)

    def set_ofield_vec(self, cell_index, vec):
        return self.impl.set_ofield_vec(cell_index, vec.impl)

    def set_stokes_vec(self, vec):
        return self.impl.set_stokes_vec(vec.impl)

    def prepare_pfield_operator(self, dt, cell_index):
        return self.impl.prepare_pfield_operator(dt, cell_index)

    def prepare_ofield_operator(self, dt, cell_index):
        return self.impl.prepare_ofield_operator(dt, cell_index)

    def set_pfield_jacobian_state(self, vec, cellindex):
        self.impl.set_pfield_jacobian_state(vec.impl, cellindex)

    def set_ofield_jacobian_state(self, vec, cellindex):
        self.impl.set_ofield_jacobian_state(vec.impl, cellindex)

    def prepare_stokes_operator(self):
        return self.impl.prepare_stokes_operator()

    def apply_inverse_pfield_operator(self, guess_vec, cell_index):
        return self.pfield_solution_space.make_array(
            [self.impl.apply_inverse_pfield_operator(guess_vec.impl, cell_index)])

    def apply_inverse_ofield_operator(self, guess_vec, cell_index):
        return self.ofield_solution_space.make_array(
            [self.impl.apply_inverse_ofield_operator(guess_vec.impl, cell_index)])

    def apply_inverse_stokes_operator(self):
        return self.stokes_solution_space.make_array([self.impl.apply_inverse_stokes_operator()])

    def apply_pfield_operator(self, U, cell_index):
        U_out = [self.impl.apply_pfield_operator(vec.impl, cell_index) for vec in U._list]
        return self.pfield_solution_space.make_array(U_out)

    def apply_ofield_operator(self, U, cell_index):
        U_out = [self.impl.apply_ofield_operator(vec.impl, cell_index) for vec in U._list]
        return self.ofield_solution_space.make_array(U_out)

    def apply_stokes_operator(self, U, mu=None):
        U_out = [self.impl.apply_stokes_operator(vec.impl) for vec in U._list]
        return self.stokes_solution_space.make_array(U_out)

    def apply_inverse_pfield_jacobian(self, V, cell_index):
        return self.pfield_solution_space.make_array(
            [self.impl.apply_inverse_pfield_jacobian(vec.impl, cell_index) for vec in V._list])

    def apply_inverse_ofield_jacobian(self, V, cell_index):
        return self.ofield_solution_space.make_array(
            [self.impl.apply_inverse_ofield_jacobian(vec.impl, cell_index) for vec in V._list])

    def apply_inverse_stokes_jacobian(self, V):
        return self.stokes_solution_space.make_array(
            [self.impl.apply_inverse_stokes_jacobian(vec.impl) for vec in V._list])

    def apply_pfield_jacobian(self, U, cell_index):
        U_out = [self.impl.apply_pfield_jacobian(vec.impl, cell_index) for vec in U._list]
        return self.pfield_solution_space.make_array(U_out)

    def apply_ofield_jacobian(self, U, cell_index):
        U_out = [self.impl.apply_ofield_jacobian(vec.impl, cell_index) for vec in U._list]
        return self.ofield_solution_space.make_array(U_out)

    def apply_stokes_jacobian(self, U, mu=None):
        U_out = [self.impl.apply_stokes_jacobian(vec.impl) for vec in U._list]
        return self.stokes_solution_space.make_array(U_out)

    def update_pfield_parameters(self, mu):
        self.impl.update_pfield_parameters(float(mu['Be']), float(mu['Ca']), float(mu['Pa']))

    def update_ofield_parameters(self, mu):
        self.impl.update_ofield_parameters(float(mu['Pa']))

    # def current_time(self):
    #     return self.impl.current_time()

    # def t_end(self):
    #     return self.impl.t_end()

    # def set_current_time(self, time):
    #     return self.impl.set_current_time(time)

    # def set_current_solution(self, vec):
    #     return self.impl.set_current_solution(vec)

    # def time_step_length(self):
    #     return self.impl.time_step_length()

    # def get_initial_values(self):
    #      return self.solution_space.make_array([self.impl.get_initial_values()])


class CellModelPfieldProductOperator(Operator):

    def __init__(self, solver):
        self.solver = solver
        self.linear = True

    def apply(self, U, mu=None, numpy=False):
        return self.solver.apply_pfield_product_operator(U, numpy=numpy)


class CellModelOfieldProductOperator(Operator):

    def __init__(self, solver):
        self.solver = solver
        self.linear = True

    def apply(self, U, mu=None):
        return self.solver.apply_ofield_product_operator(U)


class CellModelStokesProductOperator(Operator):

    def __init__(self, solver):
        self.solver = solver
        self.linear = True

    def apply(self, U, mu=None):
        return self.solver.apply_stokes_product_operator(U)


class MutableStateComponentOperator(Operator):

    mutable_state_index = (1,)
    _last_component_value = None
    _last_mu = None

    @abstractmethod
    def _change_state(self, component_value=None, mu=None):
        pass

    @abstractmethod
    def _fixed_component_apply(self, U):
        pass

    def _fixed_component_apply_inverse(self, V, least_squares=False):
        raise NotImplementedError

    def _fixed_component_jacobian(self, U):
        raise NotImplementedError

    def _set_state(self, component_value, mu):
        new_component_value = None if component_value == self._last_component_value else component_value
        new_mu = None if mu == self._last_mu else mu
        if new_component_value is not None or new_mu is not None:
            self._change_state(component_value=new_component_value, mu=new_mu)
        self._last_component_value = component_value
        self._last_mu = mu.copy() if mu is not None else None

    @property
    def fixed_component_source(self):
        subspaces = tuple(s for i, s in enumerate(self.source.subspaces) if i not in self.mutable_state_index)
        return subspaces[0] if len(subspaces) == 1 else BlockVectorSpace(subspaces)

    def apply(self, U, mu):
        assert U in self.source
        op = self.fix_component(self.mutable_state_index, U._blocks[self.mutable_state_index])
        U = U._blocks[:self.mutable_state_index] + U._blocks[self.mutable_state_index + 1:]
        if len(U) > 1:
            U = op.source.make_array(U)
        return op.apply(U, mu=mu)

    def fix_components(self, idx, U):
        if isinstance(idx, Number):
            idx = (idx,)
        if isinstance(U, VectorArray):
            U = (U,)
        assert len(idx) == len(U)
        assert all(len(u) == 1 for u in U)
        if idx != self.mutable_state_index:
            raise NotImplementedError
        assert all(u in self.source.subspaces[i] for u, i in zip(U, idx))
        return MutableStateFixedComponentOperator(self, U)


class MutableStateFixedComponentOperator(Operator):

    def __init__(self, operator, component_value):
        component_value = tuple(U.copy() for U in component_value)
        self.__auto_init(locals())
        self.source = operator.fixed_component_source
        self.range = operator.range
        self.linear = operator.linear

    def apply(self, U, mu=None):
        assert U in self.source
        self.operator._set_state(self.component_value, mu)
        return self.operator._fixed_component_apply(U)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        self.operator._set_state(self.component_value, mu)
        try:
            return self.operator._fixed_component_apply_inverse(V, least_squares=least_squares)
        except NotImplementedError:
            return super().apply_inverse(V, mu=mu, least_squares=least_squares)

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        self.operator._set_state(self.component_value, mu)
        return self.operator._fixed_component_jacobian(U, mu=mu)


class MutableStateComponentJacobianOperator(MutableStateComponentOperator):

    _last_jacobian_value = None

    @abstractmethod
    def _change_jacobian_state(self, jacobian_value):
        pass

    def _need_to_invalidate_jacobian_state(self, component_value_changed, mu_changed):
        return component_value_changed or mu_changed

    @abstractmethod
    def _fixed_component_jacobian_apply(self, U):
        pass

    def _fixed_component_jacobian_apply_inverse(self, V, least_squares=False):
        raise NotImplementedError

    def _set_state(self, component_value, mu):
        component_value_changed = component_value != self._last_component_value
        mu_changed = mu != self._last_mu
        if self._need_to_invalidate_jacobian_state(component_value_changed, mu_changed):
            self._last_jacobian_value = None
        super()._set_state(component_value, mu)

    def _set_state_jacobian(self, component_value, mu, jacobian_value):
        self._set_state(component_value, mu)
        if jacobian_value != self._last_jacobian_value:
            self._change_jacobian_state(jacobian_value)
        self._last_jacobian_value = jacobian_value

    def _fixed_component_jacobian(self, U, mu=None):
        assert len(U) == 1
        assert mu == None or mu == self._last_mu
        return MutableStateFixedComponentJacobianOperator(self, self._last_component_value, self._last_mu, U._list[0])


class MutableStateFixedComponentJacobianOperator(Operator):

    def __init__(self, operator, component_value, mu, jacobian_value):
        mu = mu.copy() if mu is not None else None
        jacobian_value = jacobian_value.copy()
        self.__auto_init(locals())
        self.source = operator.fixed_component_source
        self.range = operator.range
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        self.operator._set_state_jacobian(self.component_value, self.mu, self.jacobian_value)
        return self.operator._fixed_component_jacobian_apply(U)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        # assert mu == None or mu == self.mu # mu has to be set in the constructor
        self.operator._set_state_jacobian(self.component_value, self.mu, self.jacobian_value)
        return self.operator._fixed_component_jacobian_apply_inverse(V, least_squares=least_squares)


class CellModelPfieldOperator(MutableStateComponentJacobianOperator):

    mutable_state_index = (1, 2, 3)

    def __init__(self, solver, cell_index, dt):
        self.__auto_init(locals())
        self.linear = False
        self.source = BlockVectorSpace([
            self.solver.pfield_solution_space,
            self.solver.pfield_solution_space, self.solver.ofield_solution_space, self.solver.stokes_solution_space
        ])
        self.range = self.solver.pfield_solution_space
        self.build_parameter_type(Be=(), Ca=(), Pa=())

    def _change_state(self, component_value=None, mu=None):
        if mu is not None:
            self.solver.update_pfield_parameters(mu)
        if component_value is not None:
            self.solver.set_pfield_vec(0, component_value[0]._list[0])
            self.solver.set_ofield_vec(0, component_value[1]._list[0])
            self.solver.set_stokes_vec(component_value[2]._list[0])
        if mu is not None or component_value is not None:
            self.solver.prepare_pfield_operator(self.dt, self.cell_index)

    def _change_jacobian_state(self, jacobian_value):
        self.solver.set_pfield_jacobian_state(jacobian_value, self.cell_index)

    def _need_to_invalidate_jacobian_state(self, component_value_changed, mu_changed):
        return component_value_changed or mu_changed

    def _fixed_component_apply(self, U):
        return self.solver.apply_pfield_operator(U, self.cell_index)

    def _fixed_component_apply_inverse(self, V, least_squares=False):
        raise NotImplementedError
        assert sum(V.norm()) == 0., "Not implemented for non-zero rhs!"
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_pfield_operator(self.solver.pfield_vector(self.cell_index), self.cell_index)

    def _fixed_component_jacobian_apply(self, U):
        return self.solver.apply_pfield_jacobian(U, self.cell_index)

    def _fixed_component_jacobian_apply_inverse(self, V, least_squares=False):
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_pfield_jacobian(V, self.cell_index)


class CellModelOfieldOperator(MutableStateComponentJacobianOperator):

    mutable_state_index = (1, 2, 3)

    def __init__(self, solver, cell_index, dt):
        self.solver = solver
        self.cell_index = cell_index
        self.dt = dt
        self.linear = False
        self.source = BlockVectorSpace([
            self.solver.ofield_solution_space,
            self.solver.pfield_solution_space, self.solver.ofield_solution_space, self.solver.stokes_solution_space
        ])
        self.range = self.solver.ofield_solution_space
        self.build_parameter_type(Pa=())

    def _change_state(self, component_value=None, mu=None):
        if mu is not None:
            self.solver.update_ofield_parameters(mu)
        if component_value is not None:
            self.solver.set_pfield_vec(0, component_value[0]._list[0])
            self.solver.set_ofield_vec(0, component_value[1]._list[0])
            self.solver.set_stokes_vec(component_value[2]._list[0])
        if mu is not None or component_value is not None:
            self.solver.prepare_ofield_operator(self.dt, self.cell_index)

    def _change_jacobian_state(self, jacobian_value):
        self.solver.set_ofield_jacobian_state(jacobian_value, self.cell_index)

    def _need_to_invalidate_jacobian_state(self, component_value_changed, mu_changed):
        return component_value_changed or mu_changed

    def _fixed_component_apply(self, U):
        return self.solver.apply_ofield_operator(U, self.cell_index)

    def _fixed_component_apply_inverse(self, V, least_squares=False):
        raise NotImplementedError
        assert sum(V.norm()) == 0., "Not implemented for non-zero rhs!"
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_ofield_operator(self.solver.ofield_vector(self.cell_index), self.cell_index)

    def _fixed_component_jacobian_apply(self, U):
        return self.solver.apply_ofield_jacobian(U, self.cell_index)

    def _fixed_component_jacobian_apply_inverse(self, V, least_squares=False):
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_ofield_jacobian(V, self.cell_index)


class CellModelStokesOperator(MutableStateComponentJacobianOperator):

    mutable_state_index = (1, 2)

    def __init__(self, solver):
        self.solver = solver
        self.linear = False
        self.source = BlockVectorSpace([
            self.solver.stokes_solution_space,
            self.solver.pfield_solution_space, self.solver.ofield_solution_space
        ])
        self.range = self.solver.stokes_solution_space

    def _change_state(self, component_value=None, mu=None):
        if component_value is not None:
            self.solver.set_pfield_vec(0, component_value[0]._list[0])
            self.solver.set_ofield_vec(0, component_value[1]._list[0])
            self.solver.prepare_stokes_operator()

    def _change_jacobian_state(self, jacobian_value):
        pass

    def _need_to_invalidate_jacobian_state(self, component_value_changed, mu_changed):
        return component_value_changed

    def _fixed_component_apply(self, U):
        return self.solver.apply_stokes_operator(U)

    def _fixed_component_apply_inverse(self, V, least_squares=False):
        assert sum(V.norm()) == 0., "Not implemented for non-zero rhs!"
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_stokes_operator()

    def _fixed_component_jacobian_apply(self, U):
        return self.solver.apply_stokes_jacobian(U)

    def _fixed_component_jacobian_apply_inverse(self, V, least_squares=False):
        assert not least_squares, "Least squares not implemented!"
        return self.solver.apply_inverse_stokes_jacobian(V)


class CellModel(Model):
    def __init__(self, pfield_op, ofield_op, stokes_op,
                 initial_pfield, initial_ofield, initial_stokes,
                 dt, t_end,
                 newton_params_pfield={}, newton_params_ofield={}, newton_params_stokes={},
                 least_squares_pfield=False, least_squares_ofield=False, least_squares_stokes=False,
                 name=None):
        self.__auto_init(locals())
        self.solution_space = BlockVectorSpace(
            [pfield_op.source.subspaces[0], ofield_op.source.subspaces[0], stokes_op.source.subspaces[0]])
        self.linear = False
        self.build_parameter_type(pfield_op, ofield_op, stokes_op)

    def _solve(self, mu=None, return_output=False, return_stages=False):
        assert not return_output

        # initial values
        pfield_vecarray = self.initial_pfield.as_vector()
        ofield_vecarray = self.initial_ofield.as_vector()
        stokes_vecarray = self.initial_stokes.as_vector()

        if return_stages:
            pfield_stages = pfield_vecarray.empty()
            ofield_stages = ofield_vecarray.empty()
            stokes_stages = stokes_vecarray.empty()

        U_all = self.solution_space.make_array([pfield_vecarray, ofield_vecarray, stokes_vecarray])

        i = 0
        t = 0
        while t < self.t_end - 1e-14:
            # match saving times and t_end_ exactly
            actual_dt = min(self.dt, self.t_end - t)
            # do a timestep
            print("Current time: {}".format(t))
            pfield_vecarray, pfield_data = newton(
                self.pfield_op.fix_components((1, 2, 3), [pfield_vecarray, ofield_vecarray, stokes_vecarray]),
                self.pfield_op.range.zeros(),
                mu=mu,
                least_squares=self.least_squares_pfield,
                return_stages=return_stages,
                **self.newton_params_pfield
            )
            ofield_vecarray, ofield_data = newton(
                self.ofield_op.fix_components((1, 2, 3), [pfield_vecarray, ofield_vecarray, stokes_vecarray]),
                self.ofield_op.range.zeros(),
                mu=mu,
                least_squares=self.least_squares_ofield,
                return_stages=return_stages,
                **self.newton_params_ofield
            )
            stokes_vecarray, stokes_data = newton(
                self.stokes_op.fix_components((1, 2), [pfield_vecarray, ofield_vecarray]),
                self.stokes_op.range.zeros(),
                mu=mu,
                least_squares=self.least_squares_stokes,
                return_stages=return_stages,
                **self.newton_params_stokes
            )
            i += 1
            t += actual_dt
            U = self.solution_space.make_array([pfield_vecarray, ofield_vecarray, stokes_vecarray])
            U_all.append(U)

            if return_stages:
                pfield_stages.append(pfield_data['stages'])
                ofield_stages.append(ofield_data['stages'])
                stokes_stages.append(stokes_data['stages'])

        if return_stages:
            return U_all, (pfield_stages, ofield_stages, stokes_stages)
        else:
            return U_all

    def solve_and_check(self, mu=None):
        # initial values
        pfield_vecarray = self.initial_pfield.as_vector()
        ofield_vecarray = self.initial_ofield.as_vector()
        stokes_vecarray = self.initial_stokes.as_vector()

        U_all = self.solution_space.make_array([pfield_vecarray, ofield_vecarray, stokes_vecarray])

        i = 0
        t = 0
        while t < self.t_end - 1e-14:
            # match saving times and t_end_ exactly
            actual_dt = min(self.dt, self.t_end - t)
            # do a timestep
            #print("Current time: {}".format(t))
            pfield_fixed_op = self.pfield_op.fix_components((1, 2, 3), [pfield_vecarray, ofield_vecarray, stokes_vecarray])
            pfield_vecarray, pfield_data = newton(pfield_fixed_op, pfield_vecarray.zeros(), mu=mu,
                                                  **self.newton_params_pfield)
            residual = np.max(pfield_fixed_op.apply(pfield_vecarray, mu=mu).to_numpy())
            if residual > 1e-10:
                print("Pfield residual is ", residual)
            pfield_jacobian = pfield_fixed_op.jacobian(pfield_vecarray)
            residual = np.max((pfield_jacobian.apply_inverse(pfield_jacobian.apply(pfield_vecarray, mu=mu), mu=mu) -
                               pfield_vecarray).to_numpy())
            if residual > 1e-10:
                print("Pfield jacobian residual is ", residual)
            ofield_fixed_op = self.ofield_op.fix_components((1, 2, 3), [pfield_vecarray, ofield_vecarray, stokes_vecarray])
            ofield_vecarray, ofield_data = newton(ofield_fixed_op, ofield_vecarray.zeros(), mu=mu,
                                                  **self.newton_params_ofield)
            residual = np.max(ofield_fixed_op.apply(ofield_vecarray, mu=mu).to_numpy())
            if residual > 1e-10:
                print("Ofield residual is ", residual)
            ofield_jacobian = ofield_fixed_op.jacobian(ofield_vecarray)
            residual = np.max((ofield_jacobian.apply_inverse(ofield_jacobian.apply(ofield_vecarray, mu=mu), mu=mu) -
                               ofield_vecarray).to_numpy())
            if residual > 1e-10:
                print("Ofield jacobian residual is ", residual)
            stokes_fixed_op = self.stokes_op.fix_components((1, 2), [pfield_vecarray, ofield_vecarray])
            stokes_vecarray, stokes_data = newton(stokes_fixed_op, stokes_vecarray.zeros(), mu=mu,
                                                  **self.newton_params_stokes)
            residual = np.max(stokes_fixed_op.apply(stokes_vecarray, mu=mu).to_numpy())
            if residual > 1e-10:
                print("Stokes residual is ", residual)
            stokes_jacobian = stokes_fixed_op.jacobian(stokes_vecarray)
            residual = np.max((stokes_jacobian.apply_inverse(stokes_jacobian.apply(stokes_vecarray, mu=mu), mu=mu) -
                               stokes_vecarray).to_numpy())
            if residual > 1e-10:
                print("Stokes jacobian residual is ", residual)
            i += 1
            t += actual_dt
            U = self.solution_space.make_array([pfield_vecarray, ofield_vecarray, stokes_vecarray])
            U_all.append(U)

        return U_all



class DuneCellModel(CellModel):

    def __init__(self, solver, dt, t_end, name=None):
        pfield_op = CellModelPfieldOperator(solver, 0, dt)
        ofield_op = CellModelOfieldOperator(solver, 0, dt)
        stokes_op = CellModelStokesOperator(solver)
        initial_pfield = VectorOperator(solver.pfield_solution_space.make_array([solver.pfield_vector(0)]))
        initial_ofield = VectorOperator(solver.ofield_solution_space.make_array([solver.ofield_vector(0)]))
        initial_stokes = VectorOperator(solver.stokes_solution_space.make_array([solver.stokes_vector()]))
        super().__init__(pfield_op, ofield_op, stokes_op, initial_pfield, initial_ofield, initial_stokes,
                         dt, t_end,
                         newton_params_pfield={'atol': 1e-12, 'rtol': 1e-13},
                         newton_params_ofield={'atol': 1e-12, 'rtol': 1e-13},
                         newton_params_stokes={'atol': 1e-12, 'rtol': 1e-13},
                         name=name)
        self.solver = solver

    def visualize(self, U, prefix='cellmodel_result', subsampling=True):
        assert U in self.solution_space
        for i in range(len(U)):
            self.solver.set_pfield_vec(0, U._blocks[0]._list[i])
            self.solver.set_ofield_vec(0, U._blocks[1]._list[i])
            self.solver.set_stokes_vec(U._blocks[2]._list[i])
            self.solver.visualize(prefix, i, i, subsampling)


# class RestrictedCellModelPfieldOperator(RestrictedDuneOperatorBase):

#     linear = False

#     def __init__(self, solver, cell_index, dt, dofs):
#         self.solver = solver
#         self.dofs = dofs
#         dofs_as_list = [int(i) for i in dofs]
#         self.solver.impl.prepare_restricted_pfield_operator(dofs_as_list)
#         super(RestrictedCellModelPfieldOperator, self).__init__(solver, self.solver.impl.len_source_dofs(), len(dofs))

#     def apply(self, U, mu=None):
#         assert U in self.source
#         U = DuneXtLaListVectorSpace.from_numpy(U.to_numpy())
#         ret = [
#             DuneXtLaVector(self.solver.impl.apply_restricted_pfield_operator(u.impl, self.cell_index, self.dt)).to_numpy(True) for u in U._list
#         ]
#         return self.range.make_array(ret)


def create_and_scatter_cellmodel_parameters(comm,
                                            Re_min=1e-14,
                                            Re_max=1e-4,
                                            Fa_min=0.1,
                                            Fa_max=10,
                                            xi_min=0.1,
                                            xi_max=10):
    ''' Samples all 3 parameters uniformly with the same width and adds random parameter combinations until
        comm.Get_size() parameters are created. After that, parameter combinations are scattered to ranks. '''
    num_samples_per_parameter = int(comm.Get_size()**(1. / 3.) + 0.1)
    sample_width_Re = (Re_max - Re_min) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sample_width_Fa = (Fa_max - Fa_min) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sample_width_xi = (xi_max - xi_min) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    Re_range = np.arange(Re_min, Re_max + 1e-15, sample_width_Re)
    Fa_range = np.arange(Fa_min, Fa_max + 1e-15, sample_width_Fa)
    xi_range = np.arange(xi_min, xi_max + 1e-15, sample_width_xi)
    parameters_list = []
    for Re in Re_range:
        for Fa in Fa_range:
            for xi in xi_range:
                parameters_list.append([Re, Fa, xi])
    while len(parameters_list) < comm.Get_size():
        parameters_list.append(
            [random.uniform(Re_min, Re_max),
             random.uniform(Fa_min, Fa_max),
             random.uniform(xi_min, xi_max)])
    return comm.scatter(parameters_list, root=0)


def calculate_cellmodel_trajectory_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu):
    errs = [0.] * len(modes)
    # modes has length 2*num_cells+1
    nc = (len(modes) - 1) // 2
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    n = 0
    while not solver.finished():
        print("timestep: ", n)
        next_vectors = solver.next_n_timesteps(1, dt)
        for k in range(nc):
            res = next_vectors[k] - modes[k].lincomb(next_vectors[k].dot(
                solver.apply_pfield_product_operator(modes[k])))
            errs[k] += np.sum(res.pairwise_dot(solver.apply_pfield_product_operator(res)))
            res = next_vectors[nc + k] - modes[nc + k].lincomb(next_vectors[nc + k].dot(
                solver.apply_ofield_product_operator(modes[nc + k])))
            errs[nc + k] += np.sum(res.pairwise_dot(solver.apply_ofield_product_operator(res)))
        res = next_vectors[2 * nc] - modes[2 * nc].lincomb(next_vectors[2 * nc].dot(
            solver.apply_stokes_product_operator(modes[2 * nc])))
        errs[2 * nc] += np.sum(res.pairwise_dot(solver.apply_stokes_product_operator(res)))
        n += 1
    return errs


def calculate_mean_cellmodel_projection_errors(modes,
                                               testcase,
                                               t_end,
                                               dt,
                                               grid_size_x,
                                               grid_size_y,
                                               mu,
                                               mpi_wrapper,
                                               with_half_steps=True):
    trajectory_errs = calculate_cellmodel_trajectory_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu)
    errs = [0.] * len(modes)
    for index, trajectory_err in enumerate(trajectory_errs):
        trajectory_err = mpi_wrapper.comm_world.gather(trajectory_err, root=0)
        if mpi_wrapper.rank_world == 0:
            errs[index] = np.sqrt(np.sum(trajectory_err))
    return errs


def calculate_cellmodel_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu, mpi_wrapper, logfile=None):
    ''' Calculates projection error. As we cannot store all snapshots due to memory restrictions, the
        problem is solved again and the error calculated on the fly'''
    start = timer()
    errs = calculate_mean_cellmodel_projection_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu,
                                                      mpi_wrapper)
    elapsed = timer() - start
    if mpi_wrapper.rank_world == 0 and logfile is not None:
        logfile.write("Time used for calculating error: " + str(elapsed) + "\n")
        nc = (len(modes) - 1) // 2
        for k in range(nc):
            logfile.write("L2 error for {}-th pfield is: {}\n".format(k, errs[k]))
            logfile.write("L2 error for {}-th ofield is: {}\n".format(k, errs[nc + k]))
        logfile.write("L2 error for stokes is: {}\n".format(errs[2 * nc]))
        logfile.close()
    return errs


def get_num_chunks_and_num_timesteps(t_end, dt, chunk_size):
    num_time_steps = math.ceil(t_end / dt) + 1.
    num_chunks = int(math.ceil(num_time_steps / chunk_size))
    last_chunk_size = num_time_steps - chunk_size * (num_chunks - 1)
    assert num_chunks >= 2
    assert 1 <= last_chunk_size <= chunk_size
    return num_chunks, num_time_steps
