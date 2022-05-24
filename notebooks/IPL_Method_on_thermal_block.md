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

## First reduction globally

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

## Reduction locally

```python
from pymor.operators.constructions import ZeroOperator, LincombOperator, VectorOperator
from pymor.algorithms.projection import project
from pymor.operators.block import BlockOperator

class EllipticIPDGReductor(CoerciveRBReductor):
    def __init__(self, fom):
        self.S = fom.solution_space.empty().num_blocks
        self.fom = fom
        
        if 0:
            # this is for LincombOperator(BlockOperators)
            # pull out respective operator pairs from LincombOperator        
            self.ops_blocks = []
            if isinstance(self.fom.operator, LincombOperator):
                for ops in self.fom.operator.operators:
                    self.ops_blocks.append(ops.blocks)
            else:
                self.ops_blocks.append(self.fom.operator.blocks)

            self.rhs_blocks = []
            if isinstance(self.fom.rhs, LincombOperator):
                for ops in self.fom.rhs.operators:
                    self.rhs_blocks.append(ops)
            else:
                self.rhs_blocks.append(self.fom.rhs)

    
        self.local_bases = [fom.solution_space.empty().block(ss).empty()
                            for ss in range(self.S)]
    
    def initialize_with_global_solutions(self, us):
        assert us in self.fom.solution_space
        assert len(self.local_bases[0]) == 0
        for ss in range(self.S):
            us_block = us.block(ss)
            us_block_orth = gram_schmidt(us_block)
            self.local_bases[ss].append(us_block_orth)
            
    def add_global_solutions(self, us):
        assert us in self.fom.solution_space
        for ss in range(self.S):
            us_block = us.block(ss)
            self.local_bases[ss].append(us_block)
            # TODO: add offset
            self.local_bases[ss] = gram_schmidt(self.local_bases[ss])
    
    def add_local_solutions(self, ss, u):
        self.local_bases[ss].append(u)
        # TODO: add offset
        self.local_bases[ss] = gram_schmidt(self.local_bases[ss])
    
    def basis_length(self):
        return [len(self.local_bases[ss]) for ss in range(self.S)]
    
    def reduce(self):
        return self._reduce()
        
    def project_operators(self):
        projected_ops_blocks = []
        if 0:
            # this is for LincombOperator(BlockOperators)
            for op_blocks in self.ops_blocks:
                ops = np.empty((self.S, self.S), dtype=object)
                for ss in range(self.S):
                    for nn in range(self.S):
                        local_basis_ss = self.local_bases[ss]
                        local_basis_nn = self.local_bases[nn]
                        ops[ss][nn] = project(op_blocks[ss][nn], local_basis_ss, local_basis_nn)
                projected_ops_blocks.append(BlockOperator(ops))
            for rhs_blocks in self.rhs_blocks:
                rhs = np.empty(self.S, dtype=object)
                for ss in range(self.S):
                    local_basis_ss = self.local_bases[ss]
                    rhs_vector = VectorOperator(rhs_blocks.array.block(ss))
                    rhs_int = project(rhs_vector, local_basis_ss, None).matrix[:,0]
                    rhs[ss] = ops[ss][ss].range.make_array(rhs_int)
            projected_operator = LincombOperator(projected_ops_blocks, self.fom.operator.coefficients)
            projected_rhs = VectorOperator(projected_operator.range.make_array(rhs))
        else:
            # this is for BlockOperator(LincombOperators)
            projected_ops = np.empty((self.S, self.S), dtype=object)
            for ss in range(self.S):
                for nn in range(self.S):
                    local_basis_ss = self.local_bases[ss]
                    local_basis_nn = self.local_bases[nn]
                    if self.fom.operator.blocks[ss][nn]:
                        projected_ops[ss][nn] = project(self.fom.operator.blocks[ss][nn],
                                                        local_basis_ss, local_basis_nn)
            projected_operator = BlockOperator(projected_ops)
            
            rhs = np.empty(self.S, dtype=object)
            for ss in range(self.S):
                local_basis_ss = self.local_bases[ss]
                rhs_vector = VectorOperator(self.fom.rhs.array.block(ss))
                rhs_int = project(rhs_vector, local_basis_ss, None).matrix[:,0]
                rhs[ss] = projected_ops[ss][ss].range.make_array(rhs_int)
            projected_rhs = VectorOperator(projected_operator.range.make_array(rhs))
            
        projected_operators = {
            'operator':          projected_operator,
            'rhs':               projected_rhs,
            'products':          None,
            'output_functional': None
        }
        return projected_operators
    
    def assemble_error_estimator(self):
        return None
    
    def reconstruct(self, u_rom):
        u_ = []
        for ss in range(self.S):
            basis = self.local_bases[ss]
            u_ss = u_rom.block(ss)
            u_.append(basis.lincomb(u_ss.to_numpy()))
        return self.fom.solution_space.make_array(u_)

localized_reductor = EllipticIPDGReductor(ipl_model)
```

```python
localized_reductor.S
```

```python
localized_reductor.basis_length()
```

```python
localized_reductor.initialize_with_global_solutions(us[:3])
```

```python
localized_reductor.basis_length()
```

```python
localized_reductor.add_global_solutions(us[3:7])
```

```python
localized_reductor.basis_length()
```

```python
for i in range(0, localized_reductor.S, 2):
    u = us[7:].block(i)
    localized_reductor.add_local_solutions(i, u)
```

```python
localized_reductor.basis_length()
```

```python
# %pdb
```

```python
localized_rom = localized_reductor.reduce()

u_rom_loc = localized_rom.solve(mu)

u_rom_loc_reconstructed = localized_reductor.reconstruct(u_rom_loc)
```

```python
visualize_dd_functions(dd_grid, local_spaces, u_rom_loc_reconstructed)
```
