---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 1: creating a gmsh file

We use pyMORs `PolygonalDomain`description and `discretize_gmsh` to obtain a grid file that `gmsh` can read. **Note** that `dune-grid` can only handle `gmsh` version 2 files, we have thus installed `gmsh` version `2.16` in this virtualenv. For newer versions of `gmsh`,  you need to follow [these instructions](https://gitlab.dune-project.org/core/dune-grid/issues/85).

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

```python
from pymor.analyticalproblems.domaindescriptions import PolygonalDomain

L_shaped_domain = PolygonalDomain(points=[
    [0, 0],
    [0, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
    [1, 0],
], boundary_types={'dirichlet': [1, 2, 3, 4, 5, 6]})
```

```python
from pymor.discretizers.builtin.domaindiscretizers.gmsh import discretize_gmsh

grid, bi = discretize_gmsh(L_shaped_domain, msh_file_path='L_shaped_domain.msh')
```

# 2: discretization using pyMOR

We may use pyMOR to solve an elliptic PDE on this grid.

```python
from pymor.analyticalproblems.functions import ConstantFunction
from pymor.analyticalproblems.elliptic import StationaryProblem

problem = StationaryProblem(
    domain=L_shaped_domain,
    rhs=ConstantFunction(1, dim_domain=2),
    diffusion=ConstantFunction(1, dim_domain=2),
)
```

```python
from pymor.discretizers.builtin import discretize_stationary_cg

fom, fom_data = discretize_stationary_cg(problem, grid=grid, boundary_info=bi)
```

```python
fom.visualize(fom.solve())
```

# 3: using the gmsh grid in dune

`dune-grid` [only supports](https://gitlab.dune-project.org/core/dune-grid/issues/85) `gmsh` version 2 files, and only a subset of the specification.
This virtualenv includes the `gmsh` version 2.16 (as visible in the output of the `discretize_gmsh` command above), but we still need to clean up the mesh file for `dune-grid` to correctly parse it.
In particular, we need to remove the boundary type definition (which we do not require, we have our own boundary info), which is achieved by the following bash code (**Note** that you have to provide the same filename here as in the call to `discretize_gmsh`):

```python
# remove all lines between $PhysicalNames and $EndPhysicalNames ...
!sed '/^\$PhysicalNames/,/^\$EndPhysicalNames/{//!d;};' -i L_shaped_domain.msh
# ... and remove those two lines as well:
!sed '/^\$PhysicalNames/d' -i L_shaped_domain.msh
!sed '/^\$EndPhysicalNames/d' -i L_shaped_domain.msh
```

```python
from dune.xt.grid import make_gmsh_grid, Dim, Simplex, visualize_grid

grid = make_gmsh_grid('L_shaped_domain.msh', Dim(2), Simplex())
```

This grid can now be used as any other grid, e.g. for visualization ...

```python
_ = visualize_grid(grid)
```

... or discretization:

```python
from dune.gdt import visualize_function

from discretize_elliptic_cg import discretize_elliptic_cg_dirichlet_zero

u_h = discretize_elliptic_cg_dirichlet_zero(grid, diffusion=1, source=1)
_ = visualize_function(u_h)
```
