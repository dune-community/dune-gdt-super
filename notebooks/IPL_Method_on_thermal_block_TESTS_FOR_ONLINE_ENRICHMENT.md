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
blocks = (1,1) #(2,2)
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
# mu = [0.1, 1., 1., 1.]
mu = [1.]
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
u_ipdg = ipl_model.solve(mu)
```

```python
patch_models = return_data['patch_models']
patch_mappings = return_data['patch_mappings_to_global']
patch_mappings_to_local = return_data['patch_mappings_to_local']
```

```python
k = 9
patch_model = patch_models[k]
mapping = patch_mappings[k]
mapping_to_local = patch_mappings_to_local[k]
```

```python
u_patch = patch_model.solve(mu)
u_patch.block(1).to_numpy()
```

```python
u_restricted_to_patch = []
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping(i)
#     print(i_global)
    u_restricted_to_patch.append(u_ipdg.block(i_global))
u_restricted_to_patch = patch_model.operator.range.make_array(u_restricted_to_patch)

mu_ = ipl_model.parameters.parse(mu)
a_u_v = patch_model.operator.apply(u_restricted_to_patch, mu_)

from pymor.operators.constructions import VectorOperator
a_u_v_as_operator = VectorOperator(a_u_v)
patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
u_restricted_to_patch.block(4).to_numpy()
```

```python
(patch_model.rhs.as_range_array() - a_u_v).sup_norm()
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_.block(4).to_numpy()
```

```python
(u_restricted_to_patch - u_patch_).sup_norm()
```

```python
a_u_v_restricted_to_patch = []
mu_ = ipl_model.parameters.parse(mu)
a_u_v_global = ipl_model.operator.apply(u_ipdg, mu_)
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping(i)
    a_u_v_restricted_to_patch.append(a_u_v_global.block(i_global))
a_u_v_restricted_to_patch = patch_model.operator.range.make_array(a_u_v_restricted_to_patch)
a_u_v_as_operator = VectorOperator(a_u_v_restricted_to_patch)

patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_.sup_norm()
```

```python
from pymor.operators.constructions import LincombOperator
from pymor.operators.block import BlockOperator

u_restricted_to_patch = []
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping(i)
#     print(i_global)
    u_restricted_to_patch.append(u_ipdg.block(i_global))
u_restricted_to_patch = patch_model.operator.range.make_array(u_restricted_to_patch)

mu_ = ipl_model.parameters.parse(mu)

ops_without_outside_coupling = np.empty((len(patch_model.operator.range.subspaces), len(patch_model.operator.range.subspaces)), dtype=object)
blocks = patch_model.operator.blocks
for i in range(len(patch_model.operator.range.subspaces)):
    for j in range(len(patch_model.operator.range.subspaces)):
        if not i == j:
            if blocks[i][j]:
                # the coupling stays
                ops_without_outside_coupling[i][j] = blocks[i][j]
        else:
            # only the irrelevant couplings need to disappear
            if blocks[i][i]:
                # only works for one thermal block right now
                ss = mapping(i)
                neighborhood = [mapping(l) for l in np.arange(len(patch_model.operator.range.subspaces))]
                strings = []
                for nn in neighborhood:
                    if ss < nn:
                        strings.append(f'{ss}_{nn}')
                    else:
                        strings.append(f'{nn}_{ss}')

#                 local_ops = blocks[i][j].operators
                local_ops = [op for op in blocks[i][i].operators
                             if (op.name == 'volume' or op.name == 'boundary' or
                                 np.sum(string in op.name for string in strings))]
#                 for op in blocks[i][i].operators:
#                     print(op.name)
#                     print(np.sum(string in op.name for string in strings))
#                 print(local_ops)
                
                local_coefs = blocks[i][i].coefficients[:(len(local_ops))]
                ops_without_outside_coupling[i][i] = LincombOperator(local_ops, local_coefs)
    
operator_without_outside_couplings = BlockOperator(ops_without_outside_coupling)

a_u_v = operator_without_outside_couplings.apply(u_restricted_to_patch, mu_)

a_u_v_as_operator = VectorOperator(a_u_v)
patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_.block(0).sup_norm()
```

```python
u_patch_.sup_norm()
```

```python
def from_local_patch_to_global(u_patch):
    u_global = []
    for i in range(len(ipl_model.solution_space.subspaces)):
        i_loc = mapping_to_local(i)
        print(i_loc)
        if i_loc >= 0:
            u_global.append(u_patch.block(i_loc))
        else:
            u_global.append(ipl_model.solution_space.subspaces[i].zeros())
    return ipl_model.solution_space.make_array(u_global)

u_global_patch = from_local_patch_to_global(u_patch_)
u_global_patch.block(0)
```

```python
from dune.gdt import DiscreteFunction
from dune.gdt import visualize_function

for ss in range(8,16):
    u_list_vector_array = u_global_patch.block(ss)
    u_ss_istl = u_list_vector_array._list[0].real_part.impl
    u_ss = DiscreteFunction(local_spaces[ss], u_ss_istl, name='u_ipdg')
#     discrete_functions.append(u_ss)
    _ = visualize_function(u_ss)
    a = b
```

```python
patch_model.operator.blocks[0,0]
```

```python
macro_grid = return_data['macro_grid']
dd_grid = return_data['dd_grid']
local_spaces = return_data['local_spaces']

return_data.keys()
```

```python
# visualization
from dd_glued_visualizer import visualize_dd_functions

visualize_dd_functions(dd_grid, local_spaces, u_global_patch)
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

## Patch problems for global enrichment
