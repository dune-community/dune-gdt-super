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

# Example [WIP]: SIPDG for the instationary heat equation

## This is work in progress (WIP), still missing:

* explanations

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
grid = make_cube_grid(Dim(d), Simplex(), [0, 0], [1, 1], [2, 2])
grid.global_refine(1) # we need to refine once to obtain a symmetric grid

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

boundary_info = AllDirichletBoundaryInfo(grid)
```

# stationary data functions

```python
from dune.xt.functions import GridFunction as GF

diffusion = GF(grid, 1., dim_range=(Dim(d), Dim(d)), name='diffusion')
source = GF(grid, 1., name='source')
u_0 = GF(grid, 0., name='u_0')
```

```python
from dune.xt.grid import (
    ApplyOnCustomBoundaryIntersections,
    ApplyOnInnerIntersectionsOnce,
    DirichletBoundary,
    Walker,
)

from dune.gdt import (
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
weight = GF(grid, 1., dim_range=(Dim(d), Dim(d)), name='weight')
penalty_parameter = 16
symmetry_factor = 1

sp = make_element_and_intersection_sparsity_pattern(V_h)
m_h = MatrixOperator(grid, source_space=V_h, range_space=V_h, sparsity_pattern=sp)
m_h += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, 1)))

l_h = VectorFunctional(grid, V_h)
l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))

a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h, sparsity_pattern=sp)
a_h += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
a_h += (LocalCouplingIntersectionIntegralBilinearForm(
            LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
            + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)),
        ApplyOnInnerIntersectionsOnce(grid))
a_h += (LocalIntersectionIntegralBilinearForm(
            LocalIPDGBoundaryPenaltyIntegrand(penalty_parameter, weight)
            + LocalLaplaceIPDGDirichletCouplingIntegrand(symmetry_factor, diffusion)),
        ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))

walker = Walker(grid)
walker.append(m_h)
walker.append(a_h)
walker.append(l_h)
walker.walk()
```

$$\begin{align}
\Delta t m_h(u^{n + 1} - u^n, v) + a_h(u^{n + 1}, v) &= l_h(v)\\
m_h(u^{n + 1} - u^n, v) + \Delta t \cdot a_h(u^{n + 1}, v) &= \Delta t \cdot l_h(v)\\
m_h(u^{n + 1},v) - m_h(u^n, v) + \Delta t \cdot a_h(u^{n + 1}, v) &= \Delta t \cdot l_h(v)\\
m_h(u^{n + 1},v) + \Delta t \cdot a_h(u^{n + 1}, v) &= m_h(u^n, v) + \Delta t \cdot l_h(v)
\end{align}$$

```python
from dune.gdt import default_interpolation
from dune.xt.la import make_solver

u_h = [default_interpolation(u_0, V_h).dofs.vector]

T_end = 1
dt = 0.1
time = 0
while time < T_end + dt:
    time += dt
    u_n = u_h[-1]
    u_n_plus_one = (m_h + dt*a_h).apply_inverse(m_h.apply(u_n) + dt*l_h.vector)
    u_h.append(u_n_plus_one)    
```

```python
for ii, vec in enumerate(u_h):
    DiscreteFunction(V_h, vec, 'solution').visualize(f'solution_{ii}')
```

Start `paraview` in a terminal to visualize the time series.


# timedependent right hand side

```python
from dune.xt.functions import ExpressionFunction, ParametricExpressionFunction

u_str = '0.1*sin(20*pi*_t)*exp(-10*(x[0]*x[0]+x[1]*x[1]))'
u_0 = GF(grid,
         ExpressionFunction(dim_domain=Dim(d), variable='x', expression=u_str.replace('_t', '0'),
                            order=2, name='u_0'))

diffusion = GF(grid, 1., dim_range=(Dim(d), Dim(d)), name='diffusion')
source = GF(grid, ParametricExpressionFunction(
    dim_domain=Dim(d), variable='x', param_type={'_t': 1},
    expressions=['exp(-0.1*(x[0]*x[0]+x[1]*x[1]))*(2*pi*cos(20*pi*_t)+4*sin(20*pi*_t)-40*sin(20*pi*_t)(x[0]*x[0]+x[1]*x[1]))'],
    order=2, name='source'))
```

Use the same `m_h` and `a_h` as above, but we need to assemble `l_h` anew for each time point. We coul either directly create a new stationary `source` in each time step as an `ExpressionFunction` (like `u_0`), but we demonstrate the parametric assembly here. Computational demand is the same ...

```python
l_h = VectorFunctional(grid, V_h)
```

```python
from dune.xt.grid import ApplyOnAllElements

u_h = [default_interpolation(u_0, V_h).dofs.vector]

T_end = 1
dt = 0.1
time = 0
while time < T_end + dt:
    time += dt
    l_h.vector.set_all(0)
    l_h += (LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source)),
            {'_t': [time]},
            ApplyOnAllElements(grid))
    l_h.assemble()    
    u_n = u_h[-1]
    u_n_plus_one = (m_h + dt*a_h).apply_inverse(m_h.apply(u_n) + dt*l_h.vector)
    u_h.append(u_n_plus_one)    
```

```python
for ii, vec in enumerate(u_h):
    DiscreteFunction(V_h, vec, 'solution').visualize(f'solution_timedep_data_{ii}')
```
