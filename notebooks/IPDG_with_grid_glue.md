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
import dune.gdt
```

```python
# %pdb
# import dune.gdt._operators_interfaces_istl_1d
# import dune.gdt._ope


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
                            upper_right=omega[1], num_elements=[2, 2])
# macro_grid.global_refine(1)

macro_boundary_info = AllDirichletBoundaryInfo(macro_grid)

print(f'grid has {macro_grid.size(0)} elements, ' 
      + f'{macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')
```

Now we can use this grid as a macro grid for a dd grid.

```python
# start with no refinement on the subdomains
dd_grid = make_cube_dd_grid(macro_grid, Cube(), 1)

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
from dune.xt.grid import Dim
from dune.xt.functions import ConstantFunction, ExpressionFunction
from dune.xt.functions import GridFunction
```

```python
from dune.gdt import (BilinearForm,
                      MatrixOperator,
                      make_element_sparsity_pattern,
                      make_element_and_intersection_sparsity_pattern,
                      LocalLaplaceIntegrand,
                      LocalElementIntegralBilinearForm,
                      DirichletConstraints)
from dune.xt.grid import Walker

from pymor.bindings.dunegdt import DuneXTMatrixOperator


def assemble_local_ops(grid, space, boundary_info, d):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(
        LocalLaplaceIntegrand(GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))))
#     a_h = a_form.with(source_space=space, range_space=space)
    
    a_h.append(a_form)
    
#     dirichlet_constraints = DirichletConstraints(boundary_info, space)
    
    #walker on local grid
#     walker = Walker(grid)
#     walker.append(a_h)
#     walker.append(dirichlet_constraints)
#     walker.walk()
    
#     print('centers: ', grid.centers())
#     print(dirichlet_constraints.dirichlet_DoFs)
#     print(a_h.matrix)
    a_h.assemble()
#     dirichlet_constraints.apply(a_h.matrix)
    # TODO: first dirichlets constraints, then assemble does not work !! 
#     print(a_h.matrix.__repr__())
    op = DuneXTMatrixOperator(a_h.matrix)
    return op
```

```python
ops = np.empty((S, S), dtype=object)
```

```python
# for ss in range(S):
#     space = spaces[ss]
#     local_grid = dd_grid.local_grid(ss)
#     boundary_info = dd_grid.macro_based_boundary_info(ss, macro_boundary_info)
#     ops[ss, ss] = assemble_local_op(local_grid, space, boundary_info, d)
```

```python
from dune.gdt import LocalCouplingIntersectionIntegralBilinearForm, LocalLaplaceIPDGInnerCouplingIntegrand
from dune.gdt import LocalIPDGInnerPenaltyIntegrand
from dune.gdt import estimate_combined_inverse_trace_inequality_constant
from dune.gdt import estimate_element_to_intersection_equivalence_constant
from dune.gdt import make_coupling_sparsity_pattern

from dune.xt.grid import ApplyOnInnerIntersectionsOnce, ApplyOnCustomBoundaryIntersections

def assemble_coupling_ops(spaces, ss, nn):
    coupling_grid_ss_nn = dd_grid.coupling_grid(ss, nn)
    coupling_grid_nn_ss = dd_grid.coupling_grid(nn, ss)
    
    ss_space = spaces[ss]
    nn_space = spaces[nn]
    
    sparsity_pattern_ss_nn = make_coupling_sparsity_pattern(ss_space, nn_space, 
                                                            coupling_grid_ss_nn)
    sparsity_pattern_nn_ss = make_coupling_sparsity_pattern(nn_space, ss_space, 
                                                            coupling_grid_nn_ss)

    
    coupling_op_ss_nn = MatrixOperator(coupling_grid_ss_nn, ss_space, nn_space,
                                       sparsity_pattern_ss_nn)
    
    coupling_op_nn_ss = MatrixOperator(coupling_grid_nn_ss, nn_space, ss_space,
                                       sparsity_pattern_nn_ss)
    
    coupling_form = BilinearForm(coupling_grid)
    
    symmetry_factor = 1.
    weight = kappa
    penalty_parameter= 16
    
#     if not penalty_parameter:
        # TODO: check if we need to include diffusion for the coercivity here!
        # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
#         C_G = estimate_element_to_intersection_equivalence_constant(grid)
        # TODO: lapacke missing ! 
#         C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
#         penalty_parameter = C_G *C_M_times_1_plus_C_T
#         if symmetry_factor == 1:
#             penalty_parameter *= 4
    assert penalty_parameter > 0
    
    # grid, local_grid or coupling_grid
    diffusion = GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))
    weight = GridFunction(grid, weight, dim_range=(Dim(d), Dim(d)))
    
    coupling_integrand = LocalLaplaceIPDGInnerCouplingIntegrand(
                symmetry_factor, diffusion, weight, intersection_type=coupling)
    penalty_integrand = LocalIPDGInnerPenaltyIntegrand(
                penalty_parameter, weight, intersection_type=coupling)
    
    coupling_form += LocalCouplingIntersectionIntegralBilinearForm(coupling_integrand)
    coupling_form += LocalIntersectionIntegralBilinearForm(penalty_integrand)
    
    coupling_op.append(coupling_form)
    
    coupling_op.assemble()
    op = DuneXTMatrixOperator(coupling_op.matrix)
    return op
```

```python
def assemble_local_ops(spaces, ss, nn):
    coupling_grid_ss_nn = dd_grid.coupling_grid(ss, nn)
    coupling_grid_nn_ss = dd_grid.coupling_grid(nn, ss)
    
    ss_space = spaces[ss]
    nn_space = spaces[nn]
    
    sparsity_pattern_ss_nn = make_coupling_sparsity_pattern(ss_space, nn_space, 
                                                            coupling_grid_ss_nn)
    sparsity_pattern_nn_ss = make_coupling_sparsity_pattern(nn_space, ss_space, 
                                                            coupling_grid_nn_ss)
```

## NEW CODE !! 

```python
from dune.gdt import (BilinearForm,
                      MatrixOperator,
                      make_element_sparsity_pattern,
                      make_coupling_sparsity_pattern,
                      LocalLaplaceIntegrand,
                      LocalElementIntegralBilinearForm,
                      LocalLaplaceIPDGInnerCouplingIntegrand,
                      LocalIPDGInnerPenaltyIntegrand,
                      LocalCouplingIntersectionIntegralBilinearForm,
                      LocalIntersectionIntegralBilinearForm)
from dune.xt.grid import Walker

from pymor.bindings.dunegdt import DuneXTMatrixOperator

def assemble_subdomain_contribution(grid, space, d):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(
        LocalLaplaceIntegrand(GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))))
    
    if space.continuous:
        pass
    else:
        assert 0, "add DG contributions"
            
    ## WRITE BINDINGS FOR "WITH" METHOD
    # a_h = a_form.with(source_space=space, range_space=space)
    
    a_h.append(a_form)
    
    ## TODO: ADD RIGHT HAND SIDE
    
    #walker on local grid
    walker = Walker(grid)
    walker.append(a_h)
    # walker.append(rhs)
    walker.walk()
        
    op = DuneXTMatrixOperator(a_h.matrix)
    return op

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
    
    coupling_op_ss_ss = MatrixOperator(coupling_grid, ss_space, ss_space, coupling_sparsity_pattern)
    coupling_op_ss_nn = MatrixOperator(coupling_grid, ss_space, nn_space, coupling_sparsity_pattern)
    coupling_op_nn_ss = MatrixOperator(coupling_grid, nn_space, ss_space, coupling_sparsity_pattern)
    coupling_op_nn_nn = MatrixOperator(coupling_grid, nn_space, nn_space, coupling_sparsity_pattern)
    
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
    local_op =  assemble_subdomain_contribution(local_grid, local_space, d)
    ops[ss, ss] = local_op

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
coupling_ops
```

```python
binary_ops = [[True if op is not None else False for op in ops_] for ops_ in ops] 
for ops_ in binary_ops:
    print(ops_)
```

```python
for op in ops:
    for op_ in op:
        print(op_)
```

```python
from pymor.operators.block import BlockOperator

# TODO: use dune BINDINGS for the gdt - operator !  See PR from Felix
block_op = BlockOperator(ops)
```

```python
block_op.assemble()
```
