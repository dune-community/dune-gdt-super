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
# Attention !
# if visualization in dune is not working then
# > pip uninstall k3d
# > pip install K3D==2.6.6
```

```python
# wurlitzer: display dune's output in the notebook
# %load_ext wurlitzer

%matplotlib inline

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

# 1 Using pymor

```python
from pymor.core.logger import set_log_levels
set_log_levels({'pymor': 'WARN'})
```

```python
# export the thermal-block problem as a pymor problem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.analyticalproblems.domaindescriptions import RectDomain

d = 2
omega = ([0, 0], [1, 1])
blocks = (2, 2)
# blocks = (1, 1)
problem = thermal_block_problem(blocks)
```

```python
# add parametric rhs

from pymor.analyticalproblems.functions import LincombFunction, ConstantFunction
from pymor.parameters.functionals import ProjectionParameterFunctional

param_rhs = LincombFunction([ConstantFunction(0.5, 2), ConstantFunction(0.5, 2)],
                            [ProjectionParameterFunctional('rhs', 2, 0),
                             ProjectionParameterFunctional('rhs', 2, 1)])

problem = problem.with_(rhs = param_rhs)
```

```python
# the standard pymor way
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.discretizers.builtin.fv import discretize_stationary_fv

h = 0.01
pymor_cg, data = discretize_stationary_cg(problem, diameter=h, grid_type=RectGrid)
pymor_fv, data_ = discretize_stationary_fv(problem, diameter=h, grid_type=RectGrid)
```

```python
mu = [1., 0.1, 1., 0.1, 1., 1.]
# mu = [1.]
```

```python
pymor_grid = data['grid']
print(pymor_grid)
```

```python
# visualizing the grid

grid_enumerated = np.arange(pymor_fv.solution_space.dim)
pymor_fv.visualize(pymor_fv.solution_space.from_numpy(grid_enumerated))
```

```python
# solving the problem

u_pymor = pymor_cg.solve(mu)
pymor_cg.visualize(u_pymor)
```

```python
# solving the problem with dune cg
from pymor.discretizers.dunegdt.cg import discretize_stationary_cg as discretize_dune_cg

pymor_dune_cg, _ = discretize_dune_cg(problem, diameter=h)

u_dune_cg = pymor_dune_cg.solve(mu)
_ = pymor_dune_cg.visualize(u_dune_cg)
```

```python
# solving the problem with dune ipdg
from pymor.discretizers.dunegdt.ipdg import discretize_stationary_ipdg as discretize_dune_ipdg

pymor_dune_ipdg, _ = discretize_dune_ipdg(problem, diameter=h)

u_dune_ipdg = pymor_dune_ipdg.solve(mu)
_ = pymor_dune_ipdg.visualize(u_dune_ipdg)
```

# IPL Using DD Glued

```python
from pymor.discretizers.dunegdt.ipld3g import discretize_stationary_ipld3g
```

```python
help(discretize_stationary_ipld3g)
```

```python
# TODO: for smaller macro_diameter or more refinements, this method is very slow.
# TODO: check whether this has to do with the BlockOperator or with the penalty_parmeter or weights.

macro_diameter = np.sqrt(2)/4.
pymor_ipl_model, data = discretize_stationary_ipld3g(problem,
                                                     macro_diameter=macro_diameter,
                                                     num_local_refinements=4,
                                                     penalty_parameter=16.)
```

```python
u_pymor_ipl_dune = pymor_ipl_model.solve(mu)
```

```python
print(u_dune_cg.sup_norm())
print(u_dune_ipdg.sup_norm())
print(u_pymor_ipl_dune.sup_norm())
```

```python
macro_grid = data['macro_grid']
dd_grid = data['dd_grid']
local_spaces = data['local_spaces']

data.keys()
```

```python
# visualization
from dd_glued_visualizer import visualize_dd_functions

visualize_dd_functions(dd_grid, local_spaces, u_pymor_ipl_dune)
```

```python
np.sqrt(pymor_dune_ipdg.products['l2'].apply2(u_dune_ipdg, u_dune_ipdg))
```

```python
np.sqrt(pymor_ipl_model.products['l2'].apply2(u_pymor_ipl_dune, u_pymor_ipl_dune))
```

```python
np.sqrt(pymor_dune_ipdg.products['weighted_h1_semi_penalty'].apply2(u_dune_ipdg, u_dune_ipdg))
```

```python
np.sqrt(pymor_ipl_model.products['weighted_h1_semi_penalty'].apply2(u_pymor_ipl_dune, u_pymor_ipl_dune))
```

```python
from dune.xt.grid import visualize_grid
_ = visualize_grid(macro_grid)
```
