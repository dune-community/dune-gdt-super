import numpy as np

from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.cg import (BoundaryDirichletFunctional, L2ProductFunctionalQ1,
                                           L2ProductP1, L2ProductQ1, InterpolationOperator,
                                           BoundaryL2ProductFunctional)
from pymor.operators.constructions import (VectorOperator, ComponentProjectionOperator,
                                           LincombOperator, ZeroOperator, IdentityOperator,
                                           VectorArrayOperator)
from pymor.operators.block import BlockColumnOperator, BlockOperator
from pymor.parameters.functionals import ConstantParameterFunctional
from pymor.parameters.base import ParametricObject
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.discretizers.builtin.grids.rect import RectGrid

from pdeopt.model import QuadraticPdeoptStationaryModel
from pdeopt.discretizer import _construct_mu_bar

from pymor.discretizers.dunegdt.ipld3g import discretize_stationary_ipld3g

def discretize_quadratic_pdeopt_with_iplrb(
    problem, macro_diameter=np.sqrt(2)/4., refinements=4, weights=None,
    domain_of_interest=None, desired_temperature=None, mu_for_u_d=None,
    mu_for_tikhonov=None, pool=None, counter=None,
    store_in_tmp=False, coarse_J=False, use_fine_mesh=True,
    aFine_constructor=None, u_d=None, print_on_ranks=False):

    mu_bar = _construct_mu_bar(problem)
    # if use_fine_mesh:V
    #     primal_fom, data = discretize_stationary_cg(problem, diameter=diameter,
    #                                                 grid_type=RectGrid,
    #                                                 mu_energy_product=mu_bar)

    # discretize with ipld3g
    print('Discretizing ipld3g model ...', end='', flush=True)
    ipl_fom, data = discretize_stationary_ipld3g(problem, macro_diameter=macro_diameter,
                                                 num_local_refinements=refinements,
                                                 penalty_parameter=16.)
    num_subdomains = data['dd_grid'].num_subdomains
    print(f'done with {num_subdomains} subdomains')

    # coarse_space = coarse_model.solution_space

    if use_fine_mesh:
        grid = data['dd_grid']
    else:
        grid = coarse_grid
        data = {'grid': coarse_grid}

    d = 2

    # prepare data functions
    assert domain_of_interest is None
    domain_of_interest = domain_of_interest or ConstantFunction(1., d)

    if u_d is None:
        u_desired = ConstantFunction(desired_temperature, d) if desired_temperature is not None else None
        if mu_for_u_d is not None:
            modifified_mu = mu_for_u_d.copy()
            for key in mu_for_u_d.keys():
                if len(mu_for_u_d[key]) == 0:
                    modifified_mu.pop(key)
            if use_fine_mesh:
                u_d = ipl_fom.solve(modifified_mu)
        else:
            assert desired_temperature is not None
            u_d = InterpolationOperator(grid, u_desired).as_vector()

    # TODO: FIND THE CORRECT OPERATOR HERE !
    # TrivialOperator = IdentityOperator(ipl_fom.solution_space)
    # build a block identity operator
    local_identity_ops = np.empty((num_subdomains, num_subdomains), dtype=object)
    for I in range(num_subdomains):
        local_identity_ops[I][I] = IdentityOperator(ipl_fom.solution_space.subspaces[I])
    TrivialOperator = BlockOperator(local_identity_ops)

    # l2_u_d_squared = L2_OP.apply2(u_d, u_d)[0][0]
    l2_u_d_squared = TrivialOperator.apply2(u_d, u_d)[0][0]
    constant_part = 0.5 * l2_u_d_squared

    # assemble output functional
    from pdeopt.theta import build_output_coefficient
    if weights is not None:
        weight_for_J = weights.pop('sigma_u')
    else:
        weight_for_J = 1.
    state_functional = ConstantParameterFunctional(weight_for_J)

    if mu_for_tikhonov:
        if mu_for_u_d is not None:
            mu_for_tikhonov = mu_for_u_d
        else:
            assert isinstance(mu_for_tikhonov, dict)
    output_coefficient = build_output_coefficient(ipl_fom.parameters,
                                                  weights, mu_for_tikhonov,
                                                  None, state_functional, constant_part)

    output_functional = {}

    output_functional['output_coefficient'] = output_coefficient

    blocks_for_linear_part = np.empty(num_subdomains, dtype=object)
    for I in range(num_subdomains):
        local_linear_part = VectorArrayOperator(TrivialOperator.apply(u_d).block(I))
        blocks_for_linear_part[I] = LincombOperator([local_linear_part], [-state_functional])
    block_linear_part = BlockColumnOperator(blocks_for_linear_part)

    output_functional['linear_part'] = block_linear_part
    output_functional['d_u_linear_part'] = block_linear_part

    # output_functional['linear_part'] = LincombOperator([VectorOperator(Restricted_L2_OP.apply(u_d))],[-state_functional])      # j(.)
    # output_functional['linear_part'] = LincombOperator([VectorOperator(TrivialOperator.apply(u_d))],[-state_functional])      # j(.)


    blocks_for_bilinear_part = np.empty((num_subdomains, num_subdomains), dtype=object)
    blocks_for_d_u_bilinear_part = np.empty((num_subdomains, num_subdomains), dtype=object)
    for I in range(num_subdomains):
        blocks_for_bilinear_part[I][I] = LincombOperator(
            [TrivialOperator.blocks[I][I]], [0.5*state_functional])
        blocks_for_d_u_bilinear_part[I][I] = LincombOperator(
            [TrivialOperator.blocks[I][I]], [state_functional])
    block_bilinear_part = BlockOperator(blocks_for_bilinear_part)
    block_d_u_bilinear_part = BlockOperator(blocks_for_bilinear_part)

    output_functional['bilinear_part'] = block_bilinear_part
    output_functional['d_u_bilinear_part'] = block_d_u_bilinear_part

    # output_functional['bilinear_part'] = LincombOperator([Restricted_L2_OP],[0.5*state_functional])                              # k(.,.)
    # output_functional['bilinear_part'] = LincombOperator([TrivialOperator],[0.5*state_functional])                              # k(.,.)
    # output_functional['d_u_bilinear_part'] = LincombOperator(
        # [TrivialOperator], [state_functional])                                 # 2k(.,.)
    # output_functional['d_u_bilinear_part'] = LincombOperator(
    #     [Restricted_L2_OP], [state_functional])                                 # 2k(.,.)

    # C = domain_of_interest(grid.centers(2))  # <== these are the vertices!
    # C = np.nonzero(C)[0]
    # doI = ComponentProjectionOperator(C, Restricted_L2_OP.source)

    output_functional['sigma_u'] = state_functional
    output_functional['u_d'] = u_d
    # output_functional['DoI'] = doI

    # if use_fine_mesh:
    #     opt_product = primal_fom.energy_product                                # energy w.r.t. mu_bar (see above)
    #     primal_fom = primal_fom.with_(products=dict(opt=opt_product, **primal_fom.products))
    # else:
    #     primal_fom = None
    #     opt_product = coarse_opt_product

    # fom = primal_fom or coarse_model

    opt_product = IdentityOperator(ipl_fom.solution_space)

    pde_opt_fom = QuadraticPdeoptStationaryModel(ipl_fom, output_functional, opt_product=opt_product,
                                                 use_corrected_functional=False, adjoint_approach=False,
                                                 )
    return pde_opt_fom, data, mu_bar
