import math
import numpy as np
import random
from timeit import default_timer as timer
import weakref

from pymor.algorithms.projection import project
from pymor.core.base import abstractmethod
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.operators.constructions import (VectorOperator, ConstantOperator, LincombOperator, LinearOperator,
                                           FixedParameterOperator)
from pymor.parameters.base import Mu, Parameters, ParametricObject
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

from gdt.vectors import CommonDenseVector
import gdt.coordinatetransformedmn

from hapod.xt import DuneXtLaVector, DuneXtLaListVectorSpace

#import cvxopt
#from cvxopt import matrix as cvxmatrix

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
        self.last_mu = None
        self.testcase = testcase
        self.solution_space = DuneXtLaListVectorSpace(self.impl.get_initial_values().dim)
        self.parameters_own = {"s" : 5} if "SourceBeam" in testcase else {"s" : 4}
        self.t_end = self.impl.t_end()

    def linear(self):
        return self.impl.linear()

    def solve(self):
        times, snapshots = self.impl.solve()
        return times, self.solution_space.make_array(snapshots)

    def u_from_alpha(self, alpha_vec):
        return DuneXtLaVector(self.impl.u_from_alpha(alpha_vec.impl))

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

    def set_params(self, sigma_s_scattering=1, sigma_s_absorbing=0, sigma_t_scattering=1, sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_parameters(*mu)