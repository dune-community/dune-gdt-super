---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Example: a posteriori error estimates based on flux reconstruction

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

```python
from dune.xt.grid import Dim, Simplex, make_cube_grid, AllDirichletBoundaryInfo, visualize_grid

d = 2
grid = make_cube_grid(Dim(d), Simplex(), [-1, -1], [1, 1], [4, 4])
grid.global_refine(2)

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

boundary_info = AllDirichletBoundaryInfo(grid)
```

```python
from dune.xt.functions import ExpressionFunction, GridFunction as GF

diffusion = GF(grid, 1., dim_range=(Dim(d), Dim(d)), name='diffusion')
source = GF(grid, ExpressionFunction(
    dim_domain=Dim(d), variable='x',
    expression='0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=4, name='source'))
```

```python
from dune.xt.grid import (
    ApplyOnCustomBoundaryIntersections,
    ApplyOnInnerIntersectionsOnce,
    DirichletBoundary,
    Walker
)

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

V_h = DiscontinuousLagrangeSpace(grid, order=1)
```

```python
weight = diffusion
penalty_parameter = 16
symmetry_factor = 1

l_h = VectorFunctional(grid, V_h)
l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))

a_h = MatrixOperator(grid, V_h, V_h, make_element_and_intersection_sparsity_pattern(V_h))
a_form = BilinearForm(grid)
a_form += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
a_form += (LocalCouplingIntersectionIntegralBilinearForm(
    LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
    + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)),
           ApplyOnInnerIntersectionsOnce(grid))
a_form += (LocalIntersectionIntegralBilinearForm(
    LocalIPDGBoundaryPenaltyIntegrand(penalty_parameter, weight)
    + LocalLaplaceIPDGDirichletCouplingIntegrand(symmetry_factor, diffusion)),
           ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
a_h.append(a_form)

walker = Walker(grid)
walker.append(a_h)
walker.append(l_h)
walker.walk()
```

```python
from dune.gdt import visualize_function

u_h = DiscreteFunction(V_h, 'u_h')

a_h.apply_inverse(l_h.vector, u_h.dofs.vector)

_ = visualize_function(u_h)
```

```python
# TODO: fix this ! 

from dune.gdt import oswald_interpolation

s_h = oswald_interpolation(u_h, V_h)

_ = visualize_function(u_h - s_h, grid)
```

```python
from dune.gdt import LaplaceIpdgFluxReconstructionOperator, RaviartThomasSpace

RT_0 = RaviartThomasSpace(grid, order=0)

flux_reconstruction = LaplaceIpdgFluxReconstructionOperator(
    grid, RT_0, symmetry_factor, penalty_parameter, penalty_parameter, diffusion, weight)
# DO we need this ??
flux_reconstruction.assemble()

t_h = DiscreteFunction(RT_0, 't_h')
### FIX THIS CALL ! 
# flux_reconstruction.apply(u_h, t_h.dofs.vector)
```

```python
from dune.gdt import FiniteVolumeSpace, Operator, LocalElementBilinearFormIndicatorOperator

fv_space = FiniteVolumeSpace(grid)

eta_nc_op = Operator(grid, V_h, fv_space)
eta_nc_bil = BilinearForm(grid)
eta_nc_bil += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(weight))
# what is the correct code here ?? 
eta_nc_op += eta_nc_bil
eta_nc_2 = eta_nc_op.apply(u_h - s_h)
_ = visualize_function(eta_nc_2)
print(np.sqrt(eta_nc_2.dofs.vector.l1_norm()))
```

```python
from dune.xt.functions import divergence, ElementwiseMinimumFunction, ElementwiseDiameterFunction, inverse

h = ElementwiseDiameterFunction(grid)
min_EV = ElementwiseMinimumFunction(diffusion, search_quadrature_order=0) # we know diffusion is constant
C_P = GF(grid, 1/np.pi**2) # known for simplices

eta_r_op = Operator(grid, V_h, fv_space)
eta_r_op += LocalElementBilinearFormIndicatorOperator(
    LocalElementIntegralBilinearForm(LocalElementProductIntegrand((C_P*h*h)/min_EV)))
eta_r_2 = eta_r_op.apply(source - divergence(t_h))
print(np.sqrt(eta_r_2.dofs.vector.l1_norm()))
_ = visualize_function(eta_r_2)
```

```python
from dune.xt.functions import gradient

eta_df_op = Operator(grid, RT_0, fv_space)
eta_df_op += LocalElementBilinearFormIndicatorOperator(
    LocalElementIntegralBilinearForm(LocalElementProductIntegrand(inverse(diffusion, order=0))))
eta_df_2 = eta_df_op.apply(diffusion*gradient(u_h) + t_h)
print(eta_df_2.dofs.vector.l1_norm())
_ = visualize_function(eta_df_2)
```
