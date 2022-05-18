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

# 0. Standard FEM and Standard DG


## 0.1 using pymor

```python
from pymor.core.logger import set_log_levels
set_log_levels({'pymor': 'WARN'})
```

```python
d = 2
omega = ([0, 0], [1, 1])

# the pymor way
from pymor.analyticalproblems.functions import ConstantFunction as pymorConstantFunction
from pymor.analyticalproblems.functions import ExpressionFunction as pymorExpressionFunction
from pymor.analyticalproblems.domaindescriptions import RectDomain

pymor_omega = RectDomain([[0., 0.], [1., 1.]])
pymor_kappa = pymorConstantFunction(1., d, name='kappa')
# pymor_f = pymorExpressionFunction('exp(x[0]*x[1])', d, name='f')
pymor_f = pymorExpressionFunction('1', d, name='f')

# the dune way
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid

kappa = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='kappa')
# f = ExpressionFunction(dim_domain=Dim(d), variable='x', expression='exp(x[0]*x[1])', order=3, name='f')
f = ExpressionFunction(dim_domain=Dim(d), variable='x', expression='1', order=3, name='f')
```

```python
# the standard pymor way
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.cg import discretize_stationary_cg
from pymor.discretizers.builtin.fv import discretize_stationary_fv
from pymor.analyticalproblems.elliptic import StationaryProblem

problem = StationaryProblem(pymor_omega, pymor_f, pymor_kappa)

h = 0.2
pymor_fem_cg, data = discretize_stationary_cg(problem, diameter=h, grid_type=RectGrid)
pymor_fem_fv, data_ = discretize_stationary_fv(problem, diameter=h, grid_type=RectGrid)
```

```python
# from pymor.discretizers.builtin.cg import InterpolationOperator

# pymor_grid = data['grid']
# vis_mu = None
# diff = InterpolationOperator(pymor_grid, problem.diffusion).as_vector(vis_mu)
# rhs = InterpolationOperator(pymor_grid, problem.rhs).as_vector(vis_mu)
# pymor_fem_cg.visualize((diff, rhs))
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

u_pymor = pymor_fem_cg.solve()
pymor_fem_cg.visualize(u_pymor)
```

## 0.2 Using dune cg discretizer and get the same

```python
from dune.xt.grid import Simplex, make_cube_grid, AllDirichletBoundaryInfo, visualize_grid

grid = make_cube_grid(Dim(d), Cube(), lower_left=omega[0], upper_right=omega[1], num_elements=[8, 8])
# grid.global_refine() # we need to refine once to obtain a symmetric grid

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

boundary_info = AllDirichletBoundaryInfo(grid)

_ = visualize_grid(grid)
```

```python
from discretize_elliptic_cg import discretize_elliptic_cg_dirichlet_zero
from dune.gdt import DiscreteFunction, visualize_function

u_h = discretize_elliptic_cg_dirichlet_zero(grid, kappa, f)
_ = visualize_function(u_h)
```

## 0.2 Using DG discretizer and get the same

```python
from discretize_elliptic_ipdg import discretize_elliptic_ipdg_dirichlet_zero
```

```python
u_h = discretize_elliptic_ipdg_dirichlet_zero(grid, kappa, f,
                                              symmetry_factor=1,
                                              penalty_parameter=16, weight=1)
_ = visualize_function(u_h)
```

## 1. Creating a DD grid

Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid
from dune.xt.grid import AllDirichletBoundaryInfo
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
macro_grid = make_cube_grid(Dim(d), Cube(), lower_left=omega[0],
                            upper_right=omega[1], num_elements=[4, 4])
# macro_grid.global_refine(1)

macro_boundary_info = AllDirichletBoundaryInfo(macro_grid)

print(f'grid has {macro_grid.size(0)} elements, ' 
      + f'{macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')
```

Now we can use this grid as a macro grid for a dd grid.

```python
# start with no refinement on the subdomains
dd_grid = make_cube_dd_grid(macro_grid, Cube(), 2)

# TODO: adjust bindings to also allow for simplices !
#      Note: For this, only the correct gridprovider as return value is missing ! 

from dune.xt.grid import CouplingIntersection

coupling_intersection_type = CouplingIntersection(dd_grid)
```

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

# 2. Creating micro CG spaces


We can define cg spaces for every local grid

```python
from dune.gdt import ContinuousLagrangeSpace

S = dd_grid.num_subdomains
local_spaces = [ContinuousLagrangeSpace(dd_grid.local_grid(ss), order=1) for ss in range(S)]
local_grids = [dd_grid.local_grid(ss) for ss in range(S)]
neighbors = [dd_grid.neighbors(ss) for ss in range(S)]
```

# 3. Creating a BlockOperator for pymor

```python
from dune.xt.grid import Dim, Walker
from dune.xt.functions import ConstantFunction, ExpressionFunction
from dune.xt.functions import GridFunction

from dune.gdt import DirichletConstraints
```

```python
ops = np.empty((S, S), dtype=object)
rhs = np.empty(S, dtype=object)
```

```python
localized_dirichlet_constraints = []

for ss in range(S):
    boundary_info = dd_grid.macro_based_boundary_info(ss, macro_boundary_info)
    
    local_grid = local_grids[ss]
    local_space = local_spaces[ss]
    
    dirichlet_constraints = DirichletConstraints(boundary_info, local_space)
    
    walker = Walker(local_grid)
    walker.append(dirichlet_constraints)
    walker.walk()
    
    localized_dirichlet_constraints.append(dirichlet_constraints)
#     print(dirichlet_constraints.dirichlet_DoFs)
```

```python
from dune.gdt import (BilinearForm,
                      MatrixOperator,
                      make_element_sparsity_pattern,
                      make_coupling_sparsity_pattern,
                      make_element_and_intersection_sparsity_pattern,
                      LocalLaplaceIntegrand,
                      LocalElementIntegralBilinearForm,
                      LocalLaplaceIPDGInnerCouplingIntegrand,
                      LocalIPDGInnerPenaltyIntegrand,
                      LocalCouplingIntersectionIntegralBilinearForm,
                      LocalIntersectionIntegralBilinearForm,
                      VectorFunctional,
                      LocalElementIntegralFunctional,
                      LocalElementProductIntegrand)

from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.operators.constructions import VectorArrayOperator

def assemble_subdomain_contribution(grid, space, d, dirichlet_constraints):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(
        LocalLaplaceIntegrand(GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))))
    
    if not space.continuous:
        assert 0, "add DG contributions"
            
    ## WRITE BINDINGS FOR "WITH" METHOD
    # a_h = a_form.with(source_space=space, range_space=space)
    
    a_h.append(a_form)
    
    source = GF(grid, f)
    rhs = VectorFunctional(grid, source_space=space)
    rhs += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))

    #walker on local grid
    walker = Walker(grid)
    walker.append(a_h)
    walker.append(rhs)
    walker.walk()
    
#     print(a_h.matrix.__repr__())
    dirichlet_constraints.apply(a_h.matrix)
#     print(a_h.matrix.__repr__())
    dirichlet_constraints.apply(rhs.vector)
#     print(rhs.vector.__repr__())
    
#     a = b
    op = DuneXTMatrixOperator(a_h.matrix)
#     rhs = VectorArrayOperator(op.range.make_array([rhs.vector,]))
    rhs = op.range.make_array([rhs.vector,])
    return op, rhs

def assemble_coupling_contribution(ss, nn, ss_space, nn_space):
    coupling_grid = dd_grid.coupling_grid(ss, nn)
    
    coupling_sparsity_pattern = make_coupling_sparsity_pattern(ss_space, nn_space, coupling_grid)
        
    coupling_form = BilinearForm(coupling_grid)
    
    # TODO: FIND THE CORRECT NUMBERS HERE ! 
    symmetry_factor = 1.
    weight = kappa
    penalty_parameter= 16
    
    grid = local_grids[ss]
    diffusion = GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))
    weight = GridFunction(grid, weight, dim_range=(Dim(d), Dim(d)))
    
    coupling_integrand = LocalLaplaceIPDGInnerCouplingIntegrand(
        symmetry_factor, diffusion, weight, intersection_type=coupling_intersection_type)
    penalty_integrand = LocalIPDGInnerPenaltyIntegrand(
        penalty_parameter, weight, intersection_type=coupling_intersection_type)
    
    coupling_form += LocalCouplingIntersectionIntegralBilinearForm(coupling_integrand)
    coupling_form += LocalCouplingIntersectionIntegralBilinearForm(penalty_integrand)
    
    coupling_op_ss_ss = MatrixOperator(coupling_grid, ss_space, ss_space, make_element_sparsity_pattern(ss_space))
    coupling_op_ss_nn = MatrixOperator(coupling_grid, ss_space, nn_space, coupling_sparsity_pattern)
    coupling_op_nn_ss = MatrixOperator(coupling_grid, nn_space, ss_space, coupling_sparsity_pattern)
    coupling_op_nn_nn = MatrixOperator(coupling_grid, nn_space, nn_space, make_element_sparsity_pattern(nn_space))
    
    coupling_op_ss_ss.append(coupling_form, {}, (False, True , False, False, False, False))
    coupling_op_ss_nn.append(coupling_form, {}, (False, False, True , False, False, False))
    coupling_op_nn_ss.append(coupling_form, {}, (False, False, False, True , False, False))
    coupling_op_nn_nn.append(coupling_form, {}, (False, False, False, False, True , False))

    #walker on local grid
    walker = Walker(coupling_grid)
    walker.append(coupling_op_ss_ss)
    walker.append(coupling_op_ss_nn)
    walker.append(coupling_op_nn_ss)
    walker.append(coupling_op_nn_nn)
    walker.walk()
    
    coupling_op_ss_ss = DuneXTMatrixOperator(coupling_op_ss_ss.matrix)
    coupling_op_ss_nn = DuneXTMatrixOperator(coupling_op_ss_nn.matrix)
    coupling_op_nn_ss = DuneXTMatrixOperator(coupling_op_nn_ss.matrix)
    coupling_op_nn_nn = DuneXTMatrixOperator(coupling_op_nn_nn.matrix)
    
    return coupling_op_ss_ss, coupling_op_ss_nn, coupling_op_nn_ss, coupling_op_nn_nn
```

```python
from dune.gdt import make_element_sparsity_pattern

for ss in range(S):
    # print(f"macro element: {ss}...")
    local_space = local_spaces[ss]
    local_grid = local_grids[ss]
    dirichlet_constraints = localized_dirichlet_constraints[ss]
    local_op, local_rhs =  assemble_subdomain_contribution(local_grid, local_space, d,
                                                           dirichlet_constraints)
    ops[ss, ss] = local_op
    rhs[ss] = local_rhs
    
for ss in range(S):
    # print(f"macro element: {ss}...")
    # print(f"index: {ss}, with neigbors {dd_grid.neighbors(ss)}")
    local_space = local_spaces[ss]
    for nn in dd_grid.neighbors(ss):
        # Due to the nature of the coupling intersections, we don't have the hanging node problem. We can thus
        # treat each intersection only once.
        if ss < nn:
            neighboring_space = local_spaces[nn]
            coupling_ops = assemble_coupling_contribution(ss, nn, local_space, neighboring_space)
            
            # additional terms to diagonal
            ops[ss][ss] += coupling_ops[0]
            ops[nn][nn] += coupling_ops[3]

            # coupling terms
            if ops[ss][nn] is None:
                ops[ss][nn] = coupling_ops[1]
            else:
                ops[ss][nn] += coupling_ops[1]
            if ops[nn][ss] is None:
                ops[nn][ss] = coupling_ops[2]
            else:
                ops[nn][ss] += coupling_ops[2]
        
```

```python
# coupling_ops
```

```python
# coupling_ops[0].assemble()
```

```python
# rhs
```

```python
# binary_ops = [[True if op is not None else False for op in ops_] for ops_ in ops] 
# for ops_ in binary_ops:
#     print(ops_)
```

```python
# for op in ops:
#     for op_ in op:
#         print(op_)
```

```python
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import VectorOperator

block_op = BlockOperator(ops)
block_rhs = VectorOperator(block_op.range.make_array(rhs))
```

```python
# block_rhs
```

```python
# block_op.assemble()
```

```python
from pymor.models.basic import StationaryModel

ipdg = StationaryModel(block_op, block_rhs)

u_ipdg = ipdg.solve()
```

```python
# visualization

discrete_functions = []

for ss in range(S):
    u_list_vector_array = u_ipdg.block(ss)
    u_ss_istl = u_list_vector_array._list[0].real_part.impl
    u_ss = DiscreteFunction(local_spaces[ss], u_ss_istl, name='u_ipdg')
    discrete_functions.append(u_ss)
#     _ = visualize_function(u_ss)
```

```python
from dune.gdt import visualize_discrete_functions_on_dd_grid

_ = visualize_discrete_functions_on_dd_grid(discrete_functions, dd_grid)
```
