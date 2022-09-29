from dune.xt.grid import AllDirichletBoundaryInfo, Dim, DirichletBoundary, Walker
from dune.xt.functions import GridFunction as GF

from dune.gdt import (
    BilinearForm,
    ContinuousLagrangeSpace,
    DirichletConstraints,
    DiscreteFunction,
    LocalElementIntegralBilinearForm,
    LocalElementIntegralFunctional,
    LocalElementProductIntegrand,
    LocalLaplaceIntegrand,
    MatrixOperator,
    VectorFunctional,
    boundary_interpolation,
    make_element_sparsity_pattern,
)


def discretize_elliptic_cg_dirichlet_zero(grid, diffusion, source):

    """
    Discretizes the stationary heat equation with homogeneous Dirichlet boundary values everywhere
    with dune-gdt using continuous Lagrange finite elements.

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
    
    Returns
    -------
    u_h
        The computed solution as a dune.gdt.DiscreteFunction for postprocessing.
    """
    
    d = grid.dimension
    diffusion = GF(grid, diffusion, dim_range=(Dim(d), Dim(d)))
    source = GF(grid, source)

    boundary_info = AllDirichletBoundaryInfo(grid)
    
    V_h = ContinuousLagrangeSpace(grid, order=1)

    l_h = VectorFunctional(grid, source_space=V_h)
    l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))
    
    a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h,
                         sparsity_pattern=make_element_sparsity_pattern(V_h))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
    a_h.append(a_form)
    
    dirichlet_constraints = DirichletConstraints(boundary_info, V_h)
    
    walker = Walker(grid)
    walker.append(a_h)
    walker.append(l_h)
    walker.append(dirichlet_constraints)
    walker.walk()
    
    u_h = DiscreteFunction(V_h, name='u_h')
    dirichlet_constraints.apply(a_h.matrix, l_h.vector)
    a_h.apply_inverse(l_h.vector, u_h.dofs.vector)
    
    return u_h


def discretize_elliptic_cg_dirichlet(grid, diffusion, source, dirichlet_values):

    """
    Discretizes the stationary heat equation with non-homogeneous Dirichlet boundary values everywhere
    with dune-gdt using continuous Lagrange finite elements.

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
    source
        Function with values in R representing the Dirichlet values, anything that
        dune.xt.functions.GridFunction can handle.
    
    Returns
    -------
    u_h
        The computed solution as a dune.gdt.DiscreteFunction for postprocessing.
    """
    
    d = grid.dimension
    diffusion = GF(grid, diffusion, dim_range=(Dim(d), Dim(d)))
    source = GF(grid, source)
    dirichlet_values = GF(grid, dirichlet_values)

    boundary_info = AllDirichletBoundaryInfo(grid)
    
    V_h = ContinuousLagrangeSpace(grid, order=1)

    l_h = VectorFunctional(grid, source_space=V_h)
    l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))
    
    a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h,
                         sparsity_pattern=make_element_sparsity_pattern(V_h))
    a_h_local = LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
    a_form = BilinearForm(grid)
    a_form += a_h_local
    a_h.append(a_form)
    
    dirichlet_constraints = DirichletConstraints(boundary_info, V_h)
    dirichlet_values = boundary_interpolation(dirichlet_values, V_h, boundary_info, DirichletBoundary())
    
    walker = Walker(grid)
    walker.append(a_h)
    walker.append(l_h)
    walker.append(dirichlet_constraints)
    walker.walk()
    
    rhs_vector = l_h.vector - a_h.matrix@dirichlet_values.dofs.vector
    dirichlet_constraints.apply(a_h.matrix, rhs_vector)
    
    u_0 = a_h.apply_inverse(rhs_vector)
    
    return DiscreteFunction(V_h, u_0 + dirichlet_values.dofs.vector, name='u_h')
