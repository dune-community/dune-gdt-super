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

# Example: prolongations and computing products and norms

## This is work in progress (WIP), still missing:

* documentation and explanation
* assembled matrix as product

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

## 1: prolongation onto a finer grid

Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid
from dune.xt.functions import ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])
grid.global_refine(1)

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')
```

```python
from dune.gdt import default_interpolation, prolong, DiscontinuousLagrangeSpace

V_H = DiscontinuousLagrangeSpace(grid, order=1)

f = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='exp(x[0]*x[1])', name='f')
f_H = default_interpolation(GF(grid, f), V_H)

print(f'f_H has {len(f_H.dofs.vector)} DoFs')
```

```python
reference_grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[16, 16])
reference_grid.global_refine(1)

print(f'grid has {reference_grid.size(0)} elements')
```

```python
from dune.gdt import DiscreteFunction

V_h = DiscontinuousLagrangeSpace(reference_grid, order=1)

f_h = prolong(f_H, V_h)
print(f'f_h has {len(f_h.dofs.vector)} DoFs')
```

# 2: Computing products and norms

```python
u = GF(grid,
       ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='exp(x[0]*x[1])',
                          gradient_expressions=['x[1]*exp(x[0]*x[1])', 'x[0]*exp(x[0]*x[1])'], name='h'))
```

$$
(u, v)_{H^1} = (u, v)_{L^2} + (u, v)_{H^1_0}
$$

$$
||u||_{H^1} = \sqrt{||u||_{L^2}^2 + |u|_{H^1_0}^2}
$$


We let each product do the grid walking in `norm`

```python
from dune.gdt import (
    BilinearForm,
    LocalElementProductIntegrand,
    LocalElementIntegralBilinearForm,
    LocalLaplaceIntegrand,
)

L2_product = BilinearForm(grid)
L2_product += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, 1)))
L2_norm = L2_product.norm(u)
print(f'|| u ||_L2 = {L2_norm}')

H1_semi_product = BilinearForm(grid)
H1_semi_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, 1, dim_range=(Dim(d), Dim(d)))))
H1_semi_norm = H1_semi_product.norm(u)
print(f' | u |_H1  = {H1_semi_norm}')

H1_product = BilinearForm(grid)
H1_product += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, 1)))
H1_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, 1, dim_range=(Dim(d), Dim(d)))))
H1_norm = H1_product.norm(u)
print(f'|| u ||_H1 = {H1_norm}')
```

```python
np.allclose(np.sqrt(L2_norm**2 + H1_semi_norm**2), H1_norm)
```

- TODO: all in one grid walk?


# 3: computing errors w.r.t. a known reference solution

```python
# suppose we have a DiscreteFunction on a coarse grid, usually the solution of a discretization
# (we use the interpolation here for simplicity)

u_H = default_interpolation(u, V_H)

# prolong it onto the reference grid
u_h = prolong(u_H, V_h)

# define error function using the analytically known solution u
error = GF(grid, u - u_h)

# compute norms on reference grid
L2_product = BilinearForm(reference_grid)
L2_product += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, 1)))
print(f'|| error ||_L2 = {L2_product.norm(error)}')
```
