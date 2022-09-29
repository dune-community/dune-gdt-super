# +
from numbers import Number

from dune.xt.grid import (
    AllDirichletBoundaryInfo,
    ApplyOnCustomBoundaryIntersections,
    ApplyOnInnerIntersectionsOnce,
    Dim,
    DirichletBoundary,
    Walker,
)
from dune.xt.functions import GridFunction as GF
# -

from dune.gdt import (
    BilinearForm,
    DiscontinuousLagrangeSpace,
    DiscreteFunction,
    LocalElementIntegralBilinearForm,
    LocalElementIntegralFunctional,
    LocalElementProductIntegrand,
    LocalCouplingIntersectionIntegralBilinearForm,
    LocalIPDGBoundaryPenaltyIntegrand,
    LocalIPDGInnerPenaltyIntegrand,
    LocalIntersectionIntegralBilinearForm,
    LocalLaplaceIntegrand,
    LocalLaplaceIPDGDirichletCouplingIntegrand,
    LocalLaplaceIPDGInnerCouplingIntegrand,
    MatrixOperator,
    VectorFunctional,
    estimate_combined_inverse_trace_inequality_constant,
    estimate_element_to_intersection_equivalence_constant,
    make_element_and_intersection_sparsity_pattern,
)


def discretize_elliptic_ipdg_dirichlet_zero(grid, diffusion, source,
                                            symmetry_factor=1, penalty_parameter=None, weight=1):
    
    """
    Discretizes the stationary heat equation with homogeneous Dirichlet boundary values everywhere
    with dune-gdt using an interior penalty (IP) discontinuous Galerkin (DG) method based on first
    order Lagrange finite elements.

    The type of IPDG scheme is determined by `symmetry_factor` and `weight`:

    * with `weight==1` we obtain

    - `symmetry_factor==-1`: non-symmetric interior penalty scheme (NIPDG)
    - `symmetry_factor==0`: incomplete interior penalty scheme (IIPDG)
    - `symmetry_factor==1`: symmetric interior penalty scheme (SIPDG)

    * with `weight=diffusion`, we obtain

    - `symmetry_factor==1`: symmetric weighted interior penalty scheme (SWIPDG)
    
    Parameters
    ----------
    grid
        The grid, given as a GridProvider from dune.xt.grid.
    diffusion
        Diffusion function with values in R^{d x d}, anything that dune.xt.functions.GridFunction
        can handle.
    source
        Right hand side source function with values in R, anything that
        dune.xt.functions.GridFunction can handle.
    symmetry_factor
        Usually one of -1, 0, 1, determines the IPDG scheme (see above).
    penalty_parameter
        Positive number to ensure coercivity of the resulting diffusion bilinear form,
        e.g. 16 for simplicial non-degenerate grids in 2d and `diffusion==1`.
        Determined automatically if `None`.
    weight
        Usually 1 or diffusion, determines the IPDG scheme (see above).

    Returns
    -------
    u_h
        The computed solution as a dune.gdt.DiscreteFunction for postprocessing.
    """

    d = grid.dimension
    diffusion = GF(grid, diffusion, dim_range=(Dim(d), Dim(d)))
    source = GF(grid, source)
    weight = GF(grid, weight, dim_range=(Dim(d), Dim(d)))
    
    boundary_info = AllDirichletBoundaryInfo(grid)
    
    V_h = DiscontinuousLagrangeSpace(grid, order=1)
    if not penalty_parameter:
        # TODO: check if we need to include diffusion for the coercivity here!
        # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
        C_G = estimate_element_to_intersection_equivalence_constant(grid)
        C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
        penalty_parameter = C_G*C_M_times_1_plus_C_T
        if symmetry_factor == 1:
            penalty_parameter *= 4
    assert isinstance(penalty_parameter, Number)
    assert penalty_parameter > 0

    l_h = VectorFunctional(grid, source_space=V_h)
    l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))
    
    a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h,
                         sparsity_pattern=make_element_and_intersection_sparsity_pattern(V_h))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
    a_form += (LocalCouplingIntersectionIntegralBilinearForm(
                    LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
                    + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)),
                ApplyOnInnerIntersectionsOnce(grid))
    a_form += (LocalIntersectionIntegralBilinearForm(
                    LocalIPDGBoundaryPenaltyIntegrand(penalty_parameter, weight)),
                ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
    a_h.append(a_form)
    
    walker = Walker(grid)
    walker.append(a_h)
    walker.append(l_h)
    walker.walk()
#     print(a_h.matrix)
    a_h.assemble()
#     print(a_h.matrix)
    u_h = DiscreteFunction(V_h, name='u_h')
    a_h.apply_inverse(l_h.vector, u_h.dofs.vector)
    
    return u_h
