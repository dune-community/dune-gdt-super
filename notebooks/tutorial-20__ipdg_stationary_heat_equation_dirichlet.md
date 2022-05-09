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

# Tutorial 20 [WIP]: discontinuous IPDG for the stationary heat equation

This tutorial shows how to solve the stationary heat equation with homogeneous Dirichlet boundary conditions using interior penalty (IP) discontinuous Galerkin (DG) Finite Elmenets with `dune-gdt`.

## This is work in progress (WIP), still missing:

* mathematical theory on IPDG methods
* explanation of the IPDG implementation
* non-homonegenous Dirichlet boundary values
* Neumann boundary values
* Robin boundary values

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

```python
from dune.xt.grid import Dim
from dune.xt.functions import ConstantFunction, ExpressionFunction

d = 2
omega = ([0, 0], [1, 1])

kappa = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='kappa')
# note that we need to prescribe the approximation order, which determines the quadrature on each element
f = ExpressionFunction(dim_domain=Dim(d), variable='x', expression='exp(x[0]*x[1])', order=3, name='f')
```

```python
from dune.xt.grid import Simplex, make_cube_grid, visualize_grid

grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])
grid.global_refine(1) # we need to refine once to obtain a symmetric grid

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

_ = visualize_grid(grid)
```

# 1.9: everything in a single function

For a better overview, the above discretization code is also available in a single function in the file `discretize_elliptic_ipdg.py`.

```python
import inspect
from discretize_elliptic_ipdg import discretize_elliptic_ipdg_dirichlet_zero

print(inspect.getsource(discretize_elliptic_ipdg_dirichlet_zero))
```

```python
from dune.gdt import visualize_function

u_h = discretize_elliptic_ipdg_dirichlet_zero(
    grid, kappa, f,
    symmetry_factor=1, penalty_parameter=16, weight=1) # SIPDG scheme
```

```python
_ = visualize_function(u_h)
```

```python

```
