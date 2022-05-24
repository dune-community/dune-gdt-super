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
blocks = (2,2)
problem = thermal_block_problem(blocks)
```

```python
# the standard pymor way
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.discretizers.builtin.fv import discretize_stationary_fv

h = 0.1
pymor_fem_cg, data = discretize_stationary_cg(problem, diameter=h, grid_type=RectGrid)
pymor_fem_fv, data_ = discretize_stationary_fv(problem, diameter=h, grid_type=RectGrid)
```

```python
mu = [0.1, 1., 1., 1.]
```

```python
pymor_grid = data['grid']
print(pymor_grid)
```

```python
# visualizing the grid

grid_enumerated = np.arange(pymor_fem_fv.solution_space.dim)
pymor_fem_fv.visualize(pymor_fem_fv.solution_space.from_numpy(grid_enumerated))
```

```python
# solving the problem

u_pymor = pymor_fem_cg.solve(mu)
pymor_fem_cg.visualize(u_pymor)
```

# IPL Using DD Glued


## 1. Creating a DD grid

```python
# currently the data of the problem is communicated via a dict
# TODO: change this to the pymor way ! 
from thermal_block_for_dune import thermal_block_problem_for_dune

dune_problem = thermal_block_problem_for_dune(blocks)
```

```python
dune_problem.keys()
```

```python
from ipl_discretizer import discretize_ipl
from dune.xt.grid import Cube

ipl_model, return_data = discretize_ipl(dune_problem, omega, 
                                        num_elements       = [4, 4],
                                        global_refinements = 0,
                                        local_refinements  = 2,
                                        macro_grid_type    = Cube(),
                                        micro_grid_type    = Cube())
```

```python
macro_grid = return_data['macro_grid']
dd_grid = return_data['dd_grid']
local_spaces = return_data['local_spaces']

return_data.keys()
```

Now we can use this grid as a macro grid for a dd grid.

```python
print(dd_grid)
print(macro_grid)
```

```python
print(dd_grid.boundary_subdomains)
```

```python
from dune.xt.grid import visualize_grid
_ = visualize_grid(macro_grid)
```

```python
u_ipdg = ipl_model.solve(mu)
```

```python
# visualization
from dd_glued_visualizer import visualize_dd_functions

visualize_dd_functions(dd_grid, local_spaces, u_ipdg)
```

## Reduction globally

```python
from pymor.reductors.coercive import CoerciveRBReductor

us = ipl_model.solution_space.empty()

for mu_ in problem.parameter_space.sample_randomly(10):
    us.append(ipl_model.solve(mu_))
```

```python
from pymor.algorithms.gram_schmidt import gram_schmidt

us_orth = gram_schmidt(us)

reductor = CoerciveRBReductor(ipl_model, us_orth)

rom = reductor.reduce()
```

```python
u_rom_reconstructed = reductor.reconstruct(rom.solve(mu))
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_rom_reconstructed)
```

## Reduction locally with global solutions

```python
from iplrb_reductor import EllipticIPDGReductor

localized_reductor = EllipticIPDGReductor(ipl_model)
```

```python
print(localized_reductor.basis_length())
localized_reductor.add_global_solutions(us[:5])
print(localized_reductor.basis_length())

for i in range(0, localized_reductor.S, 2):
    u = us[5:].block(i)
    localized_reductor.add_local_solutions(i, u)

print(localized_reductor.basis_length())
```

```python
localized_rom = localized_reductor.reduce()

u_rom_loc = localized_rom.solve(mu)

u_rom_loc_reconstructed = localized_reductor.reconstruct(u_rom_loc)
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_rom_loc_reconstructed)
```
