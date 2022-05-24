import numpy as np

from dune.xt.grid import (Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid,
                          AllDirichletBoundaryInfo, Walker, CouplingIntersection,
                          ApplyOnCustomBoundaryIntersections, DirichletBoundary)

from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

from dune.gdt import (ContinuousLagrangeSpace,
                      BilinearForm,
                      MatrixOperator,
                      make_element_sparsity_pattern,
                      make_coupling_sparsity_pattern,
                      make_element_and_intersection_sparsity_pattern,
                      LocalLaplaceIntegrand,
                      LocalElementIntegralBilinearForm,
                      LocalLaplaceIPDGInnerCouplingIntegrand,
                      LocalIPDGInnerPenaltyIntegrand,
                      LocalCouplingIntersectionIntegralBilinearForm,
                      LocalIntersectionIntegralBilinearForm,
                      VectorFunctional,
                      LocalElementIntegralFunctional,
                      LocalElementProductIntegrand,
                      LocalIPDGBoundaryPenaltyIntegrand,
                      LocalLaplaceIPDGDirichletCouplingIntegrand)

from pymor.bindings.dunegdt import DuneXTMatrixOperator

from pymor.operators.constructions import VectorArrayOperator, LincombOperator, VectorOperator
from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockOperator

def discretize_ipl(problem,
                   omega              =   ([0, 0], [1, 1]),
                   num_elements       =   [4,4],
                   global_refinements =   0,
                   local_refinements  =   2,
                   macro_grid_type    =   Cube(),
                   micro_grid_type    =   Cube()):

    assert isinstance(problem, dict)
    # TODO: change this to a pymor analytical problem

    d = len(omega[0])
    macro_grid = make_cube_grid(Dim(d), macro_grid_type,
                                lower_left=omega[0], upper_right=omega[1],
                                num_elements=num_elements)

    if global_refinements:
        macro_grid.global_refine(global_refinements)

    if problem['boundary'] == 'dirichlet_zero':
        macro_boundary_info = AllDirichletBoundaryInfo(macro_grid)
    else:
        # TODO: Fill this with other boundary conditions
        raise NotImplemented

    print(f'grid has {macro_grid.size(0)} elements, ' +
          f'{macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')

    dd_grid = make_cube_dd_grid(macro_grid, micro_grid_type, local_refinements)
    # TODO: adjust bindings to also allow for simplices !
    #      Note: For this, only the correct gridprovider as return value is missing ! 

    coupling_intersection_type = CouplingIntersection(dd_grid)

    S = dd_grid.num_subdomains
    local_spaces = [ContinuousLagrangeSpace(dd_grid.local_grid(ss), order=1) for ss in range(S)]
    local_grids = [dd_grid.local_grid(ss) for ss in range(S)]
    neighbors = [dd_grid.neighbors(ss) for ss in range(S)]

    # currently loop over diffusion_expr already knowing that we want a LincombOperator
    block_ops = []

    if problem['diffusion_coefs']:
        # then we have a LincombFunction
        diffusion_expr = problem['diffusion_funcs']
        assert isinstance(diffusion_expr, list)
    else:
        diffusion_expr = [problem['diffusion_funcs']]

    for kappa in diffusion_expr:
        # TODO: make this loop more efficient to minimize the grid walks !!!
        ops = np.empty((S, S), dtype=object)

        for ss in range(S):
            # print(f"macro element: {ss}...")
            local_space = local_spaces[ss]
            local_grid = local_grids[ss]
            # TODO: minimize the amount of make_sparsity_pattern() calls !!! 
            local_op = assemble_subdomain_contribution(local_grid, local_space, d, kappa)
            ops[ss, ss] = local_op

            boundary_info = dd_grid.macro_based_boundary_info(ss, macro_boundary_info)
            boundary_op = assemble_boundary_contributions(local_grid, local_space, d,
                                                          boundary_info, kappa)
            ops[ss, ss] += boundary_op

        for ss in range(S):
            # print(f"macro element: {ss}...")
            # print(f"index: {ss}, with neigbors {dd_grid.neighbors(ss)}")
            local_space = local_spaces[ss]
            for nn in dd_grid.neighbors(ss):
                # Due to the nature of the coupling intersections,
                # we don't have the hanging node problem. We can thus
                # treat each intersection only once.
                if ss < nn:
                    neighboring_space = local_spaces[nn]
                    coupling_grid = dd_grid.coupling_grid(ss, nn)
                    coupling_ops = assemble_coupling_contribution(
                        coupling_grid, local_grid, ss, nn, local_space, neighboring_space, d, kappa,
                        coupling_intersection_type)

                    # additional terms to diagonal
                    ops[ss][ss] += coupling_ops[0]
                    ops[nn][nn] += coupling_ops[3]

                    # coupling terms
                    if ops[ss][nn] is None:
                        ops[ss][nn] = coupling_ops[1]
                    else:
                        ops[ss][nn] += coupling_ops[1]
                    if ops[nn][ss] is None:
                        ops[nn][ss] = coupling_ops[2]
                    else:
                        ops[nn][ss] += coupling_ops[2]

        block_ops.append(ops)

    block_rhs = []

    if problem['rhs_coefs']:
        # then we have a LincombFunction
        rhs_expr = problem['rhs_funcs']
        assert isinstance(rhs_expr, list)
    else:
        rhs_expr = [problem['rhs_funcs']]

    for f in rhs_expr:
        # TODO: make this loop more efficient to minimize the grid walks !!!
        rhs = np.empty(S, dtype=object)

        for ss in range(S):
            # print(f"macro element: {ss}...")
            local_space = local_spaces[ss]
            local_grid = local_grids[ss]
            local_rhs = assemble_rhs(local_grid, local_space, d, f, block_ops[0][ss,ss].range)
            rhs[ss] = local_rhs

        block_rhs.append(rhs)

    #### there are two ways how to proceed:
    if 0:
        ## FIRST LincombOperator(BlockOperators)
        diffusion_coefs = problem['diffusion_coefs']
        if diffusion_coefs:
            blocks_op = []
            for block in block_ops:
                blocks_op.append(BlockOperator(block))
            final_op = LincombOperator(blocks_op, diffusion_coefs)
        else:
            final_op = block_ops[0]

        rhs_coefs = problem['rhs_coefs']
        if rhs_coefs:
            blocks_rhs = []
            for block in block_rhs:
                assert 0, "This needs to be fixed"
                blocks_rhs.append(BlockOperator(block))
            final_rhs = LincombOperator(blocks_rhs, rhs_coefs)
        else:
            final_rhs_ = block_rhs[0]
            final_rhs = VectorOperator(final_op.range.make_array(final_rhs_))
    else:
        ### SECOND: BlockOperator(LincombOperators)
        diffusion_coefs = problem['diffusion_coefs']
        if diffusion_coefs:
            final_block_op = block_ops[0]
            # multiply the first block
            for ss in range(S):
                for nn in range(S):
                    if final_block_op[ss][nn]:
                        final_block_op[ss][nn] *= diffusion_coefs[0]
            # add the rest
            for ss in range(S):
                for nn in range(S):
                    for block_op, coef in zip(block_ops[1:], diffusion_coefs[1:]):
                        if final_block_op[ss][nn]:
                            final_block_op[ss][nn] += block_op[ss][nn] * coef

            final_op = BlockOperator(final_block_op)
        else:
            final_op = block_ops[0]

        rhs_coefs = problem['rhs_coefs']
        if rhs_coefs:
            assert 0, "This needs to be fixed"
        else:
            final_rhs_ = block_rhs[0]
            final_rhs = VectorOperator(final_op.range.make_array(final_rhs_))

    ipl_model = StationaryModel(final_op, final_rhs)

    return_data = {'macro_grid': macro_grid, 'dd_grid': dd_grid,
                   'local_spaces': local_spaces}

    return ipl_model, return_data

def assemble_rhs(grid, space, d, f, rhs_range):
    source = GF(grid, f)
    rhs = VectorFunctional(grid, source_space=space)
    rhs += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))

    #walker on local grid
    walker = Walker(grid)
    walker.append(rhs)
    walker.walk()

    rhs = rhs_range.make_array([rhs.vector,])
    return rhs

def assemble_subdomain_contribution(grid, space, d, kappa):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_and_intersection_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(
        LocalLaplaceIntegrand(GF(grid, kappa, dim_range=(Dim(d), Dim(d)))))

    if not space.continuous:
        assert 0, "add DG contributions"

    ## TODO: WRITE BINDINGS FOR "WITH" METHOD
    # a_h = a_form.with(source_space=space, range_space=space)

    a_h.append(a_form)

    #walker on local grid
    walker = Walker(grid)
    walker.append(a_h)
    walker.walk()

    op = DuneXTMatrixOperator(a_h.matrix)
    return op

def assemble_coupling_contribution(coupling_grid, grid, ss, nn, ss_space, nn_space, d, kappa,
                                   coupling_intersection_type):
    coupling_sparsity_pattern = make_coupling_sparsity_pattern(ss_space, nn_space, coupling_grid)

    coupling_form = BilinearForm(coupling_grid)

    # TODO: FIND THE CORRECT NUMBERS HERE !
    symmetry_factor = 1.
    weight = 1.
    penalty_parameter= 16

    diffusion = GF(grid, kappa, dim_range=(Dim(d), Dim(d)))
    weight = GF(grid, weight, dim_range=(Dim(d), Dim(d)))

    coupling_integrand = LocalLaplaceIPDGInnerCouplingIntegrand(
        symmetry_factor, diffusion, weight, intersection_type=coupling_intersection_type)
    penalty_integrand = LocalIPDGInnerPenaltyIntegrand(
        penalty_parameter, weight, intersection_type=coupling_intersection_type)

    coupling_form += LocalCouplingIntersectionIntegralBilinearForm(coupling_integrand)
    coupling_form += LocalCouplingIntersectionIntegralBilinearForm(penalty_integrand)

    coupling_op_ss_ss = MatrixOperator(coupling_grid, ss_space, ss_space, make_element_sparsity_pattern(ss_space))
    coupling_op_ss_nn = MatrixOperator(coupling_grid, ss_space, nn_space, coupling_sparsity_pattern)
    # TODO: transposed sparsity pattern??
    coupling_op_nn_ss = MatrixOperator(coupling_grid, nn_space, ss_space, coupling_sparsity_pattern)
    coupling_op_nn_nn = MatrixOperator(coupling_grid, nn_space, nn_space, make_element_sparsity_pattern(nn_space))

    coupling_op_ss_ss.append(coupling_form, {}, (False, True , False, False, False, False))
    coupling_op_ss_nn.append(coupling_form, {}, (False, False, True , False, False, False))
    coupling_op_nn_ss.append(coupling_form, {}, (False, False, False, True , False, False))
    coupling_op_nn_nn.append(coupling_form, {}, (False, False, False, False, True , False))

    #walker on local grid
    walker = Walker(coupling_grid)
    walker.append(coupling_op_ss_ss)
    walker.append(coupling_op_ss_nn)
    walker.append(coupling_op_nn_ss)
    walker.append(coupling_op_nn_nn)
    walker.walk()

    coupling_op_ss_ss = DuneXTMatrixOperator(coupling_op_ss_ss.matrix)
    coupling_op_ss_nn = DuneXTMatrixOperator(coupling_op_ss_nn.matrix)
    coupling_op_nn_ss = DuneXTMatrixOperator(coupling_op_nn_ss.matrix)
    coupling_op_nn_nn = DuneXTMatrixOperator(coupling_op_nn_nn.matrix)

    return coupling_op_ss_ss, coupling_op_ss_nn, coupling_op_nn_ss, coupling_op_nn_nn

def assemble_boundary_contributions(grid, space, d, boundary_info, kappa):
    # TODO: FIND THE CORRECT NUMBERS HERE ! 
    symmetry_factor = 1.
    weight = 1.
    penalty_parameter= 16
    diffusion = GF(grid, kappa, dim_range=(Dim(d), Dim(d)))
    weight = GF(grid, weight, dim_range=(Dim(d), Dim(d)))

    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_and_intersection_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += (LocalIntersectionIntegralBilinearForm(
        LocalIPDGBoundaryPenaltyIntegrand(penalty_parameter, weight) +
        LocalLaplaceIPDGDirichletCouplingIntegrand(symmetry_factor, diffusion))
            , ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
    a_h.append(a_form)

    walker = Walker(grid)
    walker.append(a_h)
    walker.walk()

    boundaryOp = DuneXTMatrixOperator(a_h.matrix)
    return boundaryOp
