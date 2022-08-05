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
# the standard pymor way
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.discretizers.builtin.fv import discretize_stationary_fv

h = 0.1
pymor_cg, data = discretize_stationary_cg(problem, diameter=h, grid_type=RectGrid)
pymor_fv, data_ = discretize_stationary_fv(problem, diameter=h, grid_type=RectGrid)
```

```python
mu = [1., 0.1, 1., 0.1]
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
from pymor.discretizers.dunegdt.cg import discretize_stationary_cg as discretize_dune_cg

pymor_dune_cg, _ = discretize_dune_cg(problem, diameter=h)
```

```python
# solving the problem with dune cg

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
pymor_ipl_model, data = discretize_stationary_ipld3g(problem, macro_diameter=0.4,
                                                     num_local_refinements=2,
                                                     penalty_parameter=16.)
```

```python
print(data.keys())
```

```python
print(data['dd_grid'].num_subdomains)
```

```python
macro_grid = data['macro_grid']
dd_grid = data['dd_grid']
local_spaces = data['local_spaces']

data.keys()
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
pymor_ipl_model.operator
```

```python
pymor_ipl_model.operator.assemble(pymor_ipl_model.parameters.parse(mu))
```

```python
u_pymor_ipl_dune = pymor_ipl_model.solve(mu)
```

```python
# visualization
from dd_glued_visualizer import visualize_dd_functions

visualize_dd_functions(dd_grid, local_spaces, u_pymor_ipl_dune)
```

## Reduction globally

```python
from pymor.reductors.coercive import CoerciveRBReductor

us = pymor_ipl_model.solution_space.empty()

for mu_ in problem.parameter_space.sample_randomly(10):
    print('.', end='', flush=True)
    us.append(pymor_ipl_model.solve(mu_))
```

```python
# For making the rhs usable in estimate_image, we need to transform it from
# a BlockOperator into a VectorOperator.
# Not working if rhs is parametric

from pymor.operators.constructions import VectorOperator

pymor_ipl_model_globally_reducable = pymor_ipl_model.with_(
    rhs = VectorOperator(pymor_ipl_model.rhs.as_vector()))
```

```python
from pymor.algorithms.gram_schmidt import gram_schmidt

us_orth = gram_schmidt(us)
print('length of the basis', len(us_orth))

reductor = CoerciveRBReductor(pymor_ipl_model_globally_reducable, us_orth)

rom = reductor.reduce()
```

```python
u_rom_reconstructed = reductor.reconstruct(rom.solve(mu))
visualize_dd_functions(dd_grid, local_spaces, u_rom_reconstructed)
```

## Reduction globally efficient reduction

```python
# To make this efficient with the global reductor, we need 
# a LincombOperator of BlockOperators in the operator

# NOT RELEVANT, since we do not intend to do this anyway
```

## Reduction locally with global solutions

```python
from pymor.reductors.coercive_ipl3dg import CoerciveIPLD3GRBReductor

localized_reductor = CoerciveIPLD3GRBReductor(pymor_ipl_model, dd_grid)
```

```python
print(localized_reductor.basis_length())
localized_reductor.add_global_solutions(us[:3])
print(localized_reductor.basis_length())

for i in range(0, localized_reductor.S, 2):
    # only add the other 7 for every second occasion
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

```python
for i in range(1, localized_reductor.S, 2):
    u = us[3:].block(i)
    localized_reductor.add_local_solutions(i, u)

print(localized_reductor.basis_length())
```

```python
localized_rom = localized_reductor.reduce()
```

```python
u_rom_loc = localized_rom.solve(mu)
u_rom_loc_reconstructed = localized_reductor.reconstruct(u_rom_loc)
visualize_dd_functions(dd_grid, local_spaces, u_rom_loc_reconstructed)
```

## Local enrichment

```python
II = 5
```

```python
phi_patch = localized_reductor.enrich_locally(II, mu, use_global_matrix=True)
phi_patch_on_global = localized_reductor.from_patch_to_global(II, phi_patch)
```

```python
visualize_dd_functions(dd_grid, local_spaces, phi_patch_on_global) #, subdomains=neighborhood)
```

```python
u_mu = localized_reductor.add_global_solutions(pymor_ipl_model.solve(mu))
localized_rom = localized_reductor.reduce()
```

```python
phi_patch = localized_reductor.enrich_locally(II, mu, use_global_matrix=True)
phi_patch_on_global = localized_reductor.from_patch_to_global(II, phi_patch)
```

```python
visualize_dd_functions(dd_grid, local_spaces, phi_patch_on_global) #, subdomains=neighborhood)
```

### Reset reductor and try efficient version

```python
localized_reductor = CoerciveIPLD3GRBReductor(pymor_ipl_model, dd_grid)
localized_reductor.add_global_solutions(us)

rom = localized_reductor.reduce()
```

```python
phi_patch = localized_reductor.enrich_locally(II, mu, use_global_matrix=False)
phi_patch_on_global = localized_reductor.from_patch_to_global(II, phi_patch)
```

```python
visualize_dd_functions(dd_grid, local_spaces, phi_patch_on_global) #, subdomains=neighborhood)
```

### first testing of the local enrichment

```python
localized_reductor = CoerciveIPLD3GRBReductor(pymor_ipl_model, dd_grid)
localized_reductor.add_global_solutions(us[:2])

localized_rom = localized_reductor.reduce()
print(localized_reductor.basis_length())
```

```python
u_rom_loc = localized_rom.solve(mu)
u_rom_loc_reconstructed = localized_reductor.reconstruct(u_rom_loc)
visualize_dd_functions(dd_grid, local_spaces, u_rom_loc_reconstructed)
```

```python
localized_reductor.enrich_all_locally(mu, use_global_matrix=False)
```

```python
localized_rom = localized_reductor.reduce()
print(localized_reductor.basis_length())
```

```python
u_rom_loc = localized_rom.solve(mu)
u_rom_loc_reconstructed = localized_reductor.reconstruct(u_rom_loc)
visualize_dd_functions(dd_grid, local_spaces, u_rom_loc_reconstructed)
```

## testing different options of local enrichment

```python
patch_model = localized_reductor.patch_models[II]
mapping_to_global = localized_reductor.patch_mappings_to_global[II]
mapping_to_local = localized_reductor.patch_mappings_to_local[II]
neighborhood = localized_reductor.neighborhoods[II]
```

```python
u_patch = patch_model.solve(mu)
u_patch_global = localized_reductor.from_patch_to_global(II, u_patch)
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_patch_global) #, subdomains=neighborhood)
```

### first, but wrong idea would be to use rhs -  a_Omega(u_h, v)

```python
u_true = pymor_ipl_model.solve(mu)

u_restricted_to_patch = []
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping_to_global(i)
    u_restricted_to_patch.append(u_pymor_ipl_dune.block(i_global))
u_restricted_to_patch = patch_model.operator.range.make_array(u_restricted_to_patch)

mu_parsed = pymor_ipl_model.parameters.parse(mu)
a_u_v = patch_model.operator.apply(u_restricted_to_patch, mu_parsed)

from pymor.operators.constructions import VectorOperator
a_u_v_as_operator = VectorOperator(a_u_v)
patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
patch_model_.rhs.assemble(mu_parsed).array.block(2).to_numpy()
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_global_ = localized_reductor.from_patch_to_global(II, u_patch_)
```

```python
u_patch_.sup_norm()
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_patch_global_) #, subdomains=neighborhood)
```

### second, correct but inefficient idea would be to use rhs -  a(u_h, v)|_Omega

```python
a_u_v_restricted_to_patch = []
mu_parsed = pymor_ipl_model.parameters.parse(mu)
a_u_v_global = pymor_ipl_model.operator.apply(u_true, mu_parsed)
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping_to_global(i)
    a_u_v_restricted_to_patch.append(a_u_v_global.block(i_global))
a_u_v_restricted_to_patch = patch_model.operator.range.make_array(a_u_v_restricted_to_patch)
a_u_v_as_operator = VectorOperator(a_u_v_restricted_to_patch)

patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_global_ = localized_reductor.from_patch_to_global(II, u_patch_)
u_patch_.sup_norm()
```

```python
# patch_model_.rhs.assemble(mu_parsed).array.block(7).to_numpy()
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_patch_global_) #, subdomains=neighborhood)
```

### third, almost correct and efficient idea would be to use rhs -  a_Omega_without_outside_coupling(u_h, v)

```python
print(neighborhood)
```

```python
from pymor.operators.constructions import LincombOperator
from pymor.operators.block import BlockOperator

u_restricted_to_patch = []
for i in range(len(patch_model.operator.range.subspaces)):
    i_global = mapping_to_global(i)
    u_restricted_to_patch.append(u_true.block(i_global))
u_restricted_to_patch = patch_model.operator.range.make_array(u_restricted_to_patch)

mu_parsed = pymor_ipl_model.parameters.parse(mu)

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
            if blocks[i][j]:
                # only works for one thermal block right now
                I = mapping_to_global(i)
                neighborhood = [mapping_to_global(l) for l in np.arange(len(patch_model.operator.range.subspaces))]
                strings = []
                for J in neighborhood:
                    if I < J:
                        strings.append(f'{I}_{J}')
                    else:
                        strings.append(f'{J}_{I}')
                local_ops, local_coefs = [], []
                for op, coef in zip(blocks[i][j].operators, blocks[i][j].coefficients):
                    if ('volume' in op.name or 'boundary' in op.name or
                                 np.sum(string in op.name for string in strings)):
                        local_ops.append(op)
                        local_coefs.append(coef)
                
                ops_without_outside_coupling[i][i] = LincombOperator(local_ops, local_coefs)

operator_without_outside_couplings = BlockOperator(ops_without_outside_coupling)

a_u_v = operator_without_outside_couplings.apply(u_restricted_to_patch, mu_parsed)

a_u_v_as_operator = VectorOperator(a_u_v)
patch_model_ = patch_model.with_(rhs = patch_model.rhs - a_u_v_as_operator)
```

```python
# patch_model_.rhs.assemble(mu_parsed).array.block(2).to_numpy()
```

```python
operator_without_outside_couplings
```

```python
patch_model.operator
```

```python
u_patch_ = patch_model_.solve(mu)
u_patch_global_ = localized_reductor.from_patch_to_global(II, u_patch_)
u_patch_.sup_norm()
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_patch_global_) #, subdomains=neighborhood)
```
