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
N = 4.
macro_diameter = np.sqrt(2)/N

pymor_ipl_model, data = discretize_stationary_ipld3g(problem, macro_diameter=macro_diameter,
                                                     num_local_refinements=2,
                                                     penalty_parameter=16.)
```

```python
u_pymor_ipl_dune = pymor_ipl_model.solve(mu)
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
from dune.xt.grid import visualize_grid
_ = visualize_grid(macro_grid)
```

## Enrichment locally with global solutions

```python
us = pymor_ipl_model.solution_space.empty()

for mu_ in problem.parameter_space.sample_randomly(5):
    print('.', end='', flush=True)
    us.append(pymor_ipl_model.solve(mu_))
```

```python
from pymor.reductors.coercive_ipl3dg import CoerciveIPLD3GRBReductor

localized_reductor = CoerciveIPLD3GRBReductor(pymor_ipl_model, dd_grid)
```

```python
print(localized_reductor.basis_length())
localized_reductor.add_global_solutions(us[:3])
print(localized_reductor.basis_length())

for i in range(0, localized_reductor.S):
    u = us[3:].block(i)
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

## Estimation


### estimator domains

```python
localized_reductor.element_patches
```

```python
localized_reductor.estimator_domains
```
