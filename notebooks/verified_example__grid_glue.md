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

## 1. Creating a DD grid

Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')
```

Now we can use this grid as a macro grid for a dd grid.

```python
dd_grid = make_cube_dd_grid(grid, 2)
```

```python
print(dd_grid)
```

We can find some binded methods in the dd_grid

```python
print([f for f in dir(dd_grid) if f[0]!='_'])
```

```python
print(dd_grid.dimension)
print(dd_grid.num_subdomains)
print(dd_grid.boundary_subdomains)
print(dd_grid.neighbors(1))
```

We can also construct a local grid on a subdomain. This grid can be treated completely analog to the standard grid

```python
local_grid = dd_grid.local_grid(0)
```

```python
print([f for f in dir(local_grid) if f[0]!='_'])
```

```python
print(local_grid.centers())
print(local_grid.max_level)
print(local_grid.size(0))
```

We can define cg spaces for every local grid

```python
from dune.gdt import ContinuousLagrangeSpace

micro_spaces = []
local_grids = []
for lg_idx in range(dd_grid.num_subdomains):
    local_grid = dd_grid.local_grid(lg_idx)
    local_gp = ContinuousLagrangeSpace(local_grid, order=1)
    micro_spaces.append(local_gp)
    local_grids.append(local_grid)
```

```python
micro_spaces
```

## 2. Visualizing the grid

```python
from dune.xt.grid import visualize_grid

_ = visualize_grid(local_grid)
```

```python
_ = visualize_grid(dd_grid.local_grid(6))
```

```python
_ = visualize_grid(grid)
```

```python
print(dd_grid.boundary_subdomains)
```

```python
print(dd_grid.neighbors(1))
```

## 3. Visualizing and interpolation of functions


### 2.1 some functions

Lets define some expression functions.

```python
f1 = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='(0.5 - x[0])^2 * (0.5 - x[1])^2', 
                         name='f1')
f2 = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='x[0]*x[1]', name='f2')
```

and some discrete function on the macro grid

```python
from dune.gdt import DiscontinuousLagrangeSpace, DiscreteFunction

V_H = DiscontinuousLagrangeSpace(grid, order=1)
v_H = DiscreteFunction(V_H, name='v_h')
```

Lets visualize f1 and f2 on the macro grid

```python
from dune.gdt import visualize_function

_ = visualize_function(f1, grid, subsampling=True)
```

```python
_ = visualize_function(f2, grid, subsampling=True)
```

### 2.2 Interpolate GridFunctions

```python
from dune.gdt import default_interpolation
from dune.xt.functions import GridFunction

f1_grid = GridFunction(grid, f1)
f2_grid = GridFunction(grid, f2)

f1_h = default_interpolation(f1_grid, V_H)
f2_h = default_interpolation(f2_grid, V_H)
```

```python
_ = visualize_function(f1_h, subsampling=True) 
```

```python
_ = visualize_function(f2_h, subsampling=False)
```

Interpolate f1 and f2 to two subdomains

```python
f1_grid_1 = GridFunction(local_grids[1], f1)
f2_grid_1 = GridFunction(local_grids[1], f2)

f1_h_1 = default_interpolation(f1_grid_1, micro_spaces[1])
f2_h_1 = default_interpolation(f2_grid_1, micro_spaces[1])
```

```python
_ = visualize_function(f1_h_1, subsampling=True)
```

```python
_ = visualize_function(f2_h_1, subsampling=True)
```

```python
f1_grid_4 = GridFunction(local_grids[4], f1)
f2_grid_4 = GridFunction(local_grids[4], f2)

f1_h_4 = default_interpolation(f1_grid_4, micro_spaces[4])
f2_h_4 = default_interpolation(f2_grid_4, micro_spaces[4])
```

```python
_ = visualize_function(f1_h_4, subsampling=True)
```

```python
_ = visualize_function(f2_h_4, subsampling=True)
```

```python
from dune.xt.functions import visualize_function_on_dd_grid

_ = visualize_function_on_dd_grid(f1, dd_grid, [7])
```

```python
_ = visualize_function_on_dd_grid(f1, dd_grid)
```

```python
_ = visualize_function_on_dd_grid(f2, dd_grid)
```

```python
# TODO: how can we make this work?
_ = visualize_function_on_dd_grid(f1, dd_grid, [1,2])
```
