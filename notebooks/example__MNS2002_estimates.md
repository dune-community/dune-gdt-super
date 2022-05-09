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

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

# 1: inspect grid properties and relations between elements and intersections

```python
from dune.xt.grid import Dim, Simplex, make_cube_grid, AllDirichletBoundaryInfo, visualize_grid

d = 2
grid = make_cube_grid(Dim(d), Simplex(), [-1, -1], [1, 1], [1, 1])
grid.global_refine(1)

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

boundary_info = AllDirichletBoundaryInfo(grid)

_ = visualize_grid(grid)
```

```python
intersection_centers = np.array(grid.centers(1), copy=False)

inner_intersections = np.array(grid.inner_intersection_indices(), copy=False)
print(f'grid has {len(inner_intersections)} inner intersections')
print(f'  with indices {inner_intersections}')
print(f'  and centers:')
print(intersection_centers[inner_intersections])
```

```python
from dune.xt.grid import ApplyOnCustomBoundaryIntersections, DirichletBoundary

dirichlet_intersections = np.array(
    grid.boundary_intersection_indices(intersection_filter=
        ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary())), copy=False)
print(f'grid has {len(dirichlet_intersections)} Dirichlet intersections')
print(f'  with indices {dirichlet_intersections}')
print(f'  and centers:')
print(intersection_centers[dirichlet_intersections])
```

```python
element_centers = np.array(grid.centers(0), copy=False)

inner_element_indices = np.array(grid.inside_element_indices(), copy=False)
print(f'index of each intersections inside element: {inner_element_indices}')
print('this means that')
for intersection_index, element_index in enumerate(inner_element_indices):
    print(f'  intersection {intersection_index} with center {intersection_centers[intersection_index]}')
    print(f'    has inside element {element_index} with center {element_centers[element_index]}')
```

```python
outside_element_indices = np.array(grid.outside_element_indices(), copy=False)
print(f'index of each intersections outside element: {outside_element_indices}')
```

The large numbers are invalid indices and represent intersections which do not have an outside element, e.g. boundary intersections. We should thus restrict the lookup to inner intersections:

```python
print(f'index of each inner intersections outside element: {outside_element_indices[inner_intersections]}')
print('this means that')
for intersection_index, element_index in enumerate(outside_element_indices[inner_intersections]):
    print(f'  (inner) intersection {intersection_index} with center {intersection_centers[intersection_index]}')
    print(f'    has outside element {element_index} with center {element_centers[element_index]}')
```

# 2: solving an elliptic PDE

```python
from dune.xt.functions import ExpressionFunction, GridFunction as GF

A = GF(grid, 1., dim_range=(Dim(d), Dim(d)), name='A')
f = GF(grid,
       ExpressionFunction(dim_domain=Dim(d), variable='x', expression='exp(x[0]*x[1])', order=7, name='f'))
```

```python
from dune.xt.grid import AllDirichletBoundaryInfo, Dim, Walker

from dune.gdt import (
    ContinuousLagrangeSpace,
    DirichletConstraints,
    DiscreteFunction,
    LocalElementIntegralBilinearForm,
    LocalElementIntegralFunctional,
    LocalElementProductIntegrand,
    LocalLaplaceIntegrand,
    MatrixOperator,
    VectorFunctional,
    make_element_sparsity_pattern,
)

V_h = ContinuousLagrangeSpace(grid, order=1)
```

```python
l_h = VectorFunctional(grid, source_space=V_h)
l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(f))

a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h,
                     sparsity_pattern=make_element_sparsity_pattern(V_h))
a_h += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(A))

dirichlet_constraints = DirichletConstraints(boundary_info, V_h)

walker = Walker(grid)
walker.append(a_h)
walker.append(l_h)
walker.append(dirichlet_constraints)
walker.walk()

dirichlet_constraints.apply(a_h.matrix, l_h.vector)
```

```python
from dune.gdt import visualize_function

u_h = DiscreteFunction(V_h, 'u_h')

a_h.apply_inverse(l_h.vector, u_h.dofs.vector)

_ = visualize_function(u_h)
```

# 3: computing the a posterior error estimates and indicators from [MNS2002]

We consider the set of all faces (or sides) of the grid, $\mathcal{F}_h$, with a face denoted by $\Gamma \in \mathcal{F}_h$.
Each face $\Gamma \in \mathcal{F}_h$ is either

* an inner face, i.e. $\Gamma = \overline{\partial K^- \cap \partial K^+}$ for two grid elements $K^-, K^+ \in \mathcal{T}_h$, or
* a boundary face, i.e. $\Gamma = \overline{\partial K^- \cap \partial\Omega}$ for a grid element $K^- \in \mathcal{T}_h$

and has a

* unit outer normal $n_\Gamma \in \mathbb{R}^d$ poiting away from $K^-$.

We call $K^-$ the *inner* element and $K^+$ the outer element.

The local face indicator in [MNS2002, (2.6)] is defined by

$$\begin{align}
\eta_\Gamma^2 := \underbrace{\big\| \sqrt{h_\gamma}\; [[A \nabla u_h]]\cdot n_\Gamma \big\|_{L^2(\Gamma)}^2}_{:= \eta_{\Gamma\text{, flux}}^2} + \underbrace{\|h\,f\|_{L^2(\omega_\Gamma)}^2}_{=: \eta_{\Gamma, f}^2}
\end{align}$$

where $h_\Gamma := \text{diam}(\Gamma)$ for all $\Gamma \in \mathcal{F}_h$, $h|_K := \text{diam}(K)$ for all $K \in \mathcal{T}_h$ and

* the jump of a vector valued function $v: \Omega \to \mathbb{R}^d$ over a face $\Gamma$ defined by

  $$\begin{align}
  [[v]] := \begin{cases}v^- - v^+, &\Gamma\text{ is an inner intersection and}\\v^-,&\Gamma\text{ is a boundary intersection}\end{cases}
  \end{align}$$
  
  where $v^- := v|_{K^-}$ and $v^+ := v|_{K^+}$; and

* the patch

  $$\begin{align}
  \omega_\Gamma := \begin{cases}K^- \cup K^+, &\Gamma\text{ is an inner intersection and}\\K^-,&\Gamma\text{ is a boundary intersection}\end{cases}
  \end{align}$$
  
To compute the first contribution to the indicator, we use an operator mapping the flux $A \nabla u_h$ to the vector
$\underline{\eta_\text{flux}} \in \mathbb{R}^{|\mathcal{F}_h|}$, where the $i$-the entry of the vector is the squared indicator contribution associated with the respective intersection, i.e.

$$\begin{align}
\underline{\eta_\text{flux}}_i := \eta_{\Gamma\text{, flux}}^2&&\text{with }\Gamma\text{ the intersection with global index }i.
\end{align}$$

We use a `RaviartThomasSpace` to represent the flux space (only for its dimensions, the flux need not be an actual element of the space), and `FiniteVolumeSkeletonSpace` to represent numbers associated with intersections.

```python
from dune.xt.grid import ApplyOnInnerIntersectionsOnce, ApplyOnCustomBoundaryIntersections, DirichletBoundary

from dune.xt.functions import gradient

from dune.gdt import (
    FiniteVolumeSkeletonSpace,
    LocalCouplingIntersectionIntegralBilinearForm,
    LocalCouplingIntersectionBilinearFormIndicatorOperator,
    LocalIntersectionIntegralBilinearForm,
    LocalIntersectionBilinearFormIndicatorOperator,
    LocalInnerJumpIntegrand,
    LocalBoundaryJumpIntegrand,
    Operator,
    RaviartThomasSpace,
)

flux_space = RaviartThomasSpace(grid, order=0)
flux = A*gradient(u_h)

intersection_indices = FiniteVolumeSkeletonSpace(grid)

eta_flux_op = Operator(grid, flux_space, intersection_indices)
eta_flux_op += (
    LocalCouplingIntersectionBilinearFormIndicatorOperator(
        LocalCouplingIntersectionIntegralBilinearForm(
            LocalInnerJumpIntegrand(grid, dim_range=Dim(d), weighted_by_intersection_diameter=True))),
    ApplyOnInnerIntersectionsOnce(grid))
eta_flux_op += (
    LocalIntersectionBilinearFormIndicatorOperator(
        LocalIntersectionIntegralBilinearForm(
            LocalBoundaryJumpIntegrand(grid, dim_range=Dim(d), weighted_by_intersection_diameter=True))),
    ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
eta_flux_2 = np.array(eta_flux_op.apply(flux).dofs.vector, copy=False)
print(f'flux jump indicators = {eta_flux_2}')
```

```python
# individual entries may contain negative values due to numerical inaccuracies, which we set to zero here
negative_entries = np.where(eta_flux_2 < 0)[0]
if len(negative_entries)> 0:
    eta_flux_2[negative_entries] = np.zeros(len(negative_entries))
    print(f'flux jump indicators (cleaned up) = {eta_flux_2}')
del negative_entries
```

For the second contribution we first compute $\|h f\|_{L^2(K)}^2$ for all $K \in \mathcal{T}_h$ and combine them to obtain the intersection patch indicators.

```python
from dune.xt.functions import ElementwiseDiameterFunction

from dune.gdt import (
    LocalElementBilinearFormIndicatorOperator,
    FiniteVolumeSpace,
    default_interpolation,
)

element_indices = FiniteVolumeSpace(grid)
h = ElementwiseDiameterFunction(grid)

eta_f_op = Operator(grid, V_h, element_indices)
eta_f_op += LocalElementBilinearFormIndicatorOperator(
    LocalElementIntegralBilinearForm(LocalElementProductIntegrand(weight=GF(grid, 1))))
eta_f_2_per_element = np.array(eta_f_op.apply(h*f).dofs.vector, copy=False)
print(eta_f_2_per_element)
```

```python
eta_f_2_per_intersection = np.zeros(intersection_indices.num_DoFs)

eta_f_2_per_intersection += eta_f_2_per_element[inner_element_indices]
eta_f_2_per_intersection[inner_intersections] += eta_f_2_per_element[outside_element_indices[inner_intersections]]
```

We can now obtain the indicators simply by combining the two contributions.

```python
eta_2_per_intersection = eta_flux_2 + eta_f_2_per_intersection
```

Since `eta_2_per_intersection` contains the squared contributions, we obtain the estimate as the square root of the sum.

```python
eta = np.sqrt(np.linalg.norm(eta_2_per_intersection, ord=1))

print(f'estimated error: {eta}')
```

# 4: data oscialltion

We compute the data oscillation on a finer grid, for visualization. **Note** that all quantities from above are now invalid!

```python
grid.global_refine(4)

V_h = ContinuousLagrangeSpace(grid, order=1)
```

```python
# f_h is the piecewise constant average value on each element
p0_space = FiniteVolumeSpace(grid)
f_h = default_interpolation(f, p0_space)

oscillation_op = Operator(grid, V_h, p0_space)
oscillation_op += LocalElementBilinearFormIndicatorOperator(
    LocalElementIntegralBilinearForm(
        LocalElementProductIntegrand(weight=GF(grid, 1))))
osc_2 = oscillation_op.apply(h*(f - f_h))

# osc_2 is a discrete function, where the entries of its DoF vector
# correspond to the squared data oscillation on each grid element.
# Since the information is element-based (and not intersection-based),
# we can easily visualize it:

_ = visualize_function(osc_2)
```

```python
# we obtain ||h(f - f_h)||_{L^2(\Omega)} by the L^1-norm since the indicators are already squared
print(np.sqrt(osc_2.dofs.vector.l1_norm()))
```
