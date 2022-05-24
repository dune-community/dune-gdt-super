# the dune way
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF
from dune.xt.functions import IndicatorFunction__2d_to_1x1 as IndicatorFunction
from dune.xt.grid import Dim

from pymor.parameters.functionals import ProjectionParameterFunctional

from itertools import product

def thermal_block_problem_for_dune(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.
    The problem is to solve the elliptic equation ::
      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1
    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block i with value μ_i. ::
           ----------------------------
           |        |        |        |
           |  μ_4   |  μ_5   |  μ_6   |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_1   |  μ_2   |  μ_3   |
           |        |        |        |
           ----------------------------
    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_i is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional('diffusion',
                                             size=num_blocks[0]*num_blocks[1],
                                             index=ix + iy*num_blocks[0],
                                             name=f'diffusion_{ix}_{iy}')

    def diffusion_function_factory(ix, iy):
        dx = 1. / num_blocks[0]
        dy = 1. / num_blocks[1]
        X = [ix * dx, iy * dy]
        Y = [(ix+1) * dx, (iy+1) * dy]
        function = IndicatorFunction([(X, Y, [1.])])
        return function

    d = len(num_blocks)
    rhs_funcs = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='f')
    rhs_coefs = None

    diffusion_funcs = [diffusion_function_factory(ix, iy)
                             for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))]
    diffusion_coefs = [parameter_functional_factory(ix, iy)
                                for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))]

    return {'diffusion_funcs' : diffusion_funcs,
            'diffusion_coefs' : diffusion_coefs,
            'rhs_funcs' : rhs_funcs,
            'rhs_coefs' : rhs_coefs,
            'boundary': 'dirichlet_zero'}
