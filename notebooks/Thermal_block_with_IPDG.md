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
mu = [0.1,1.,1.,1.]
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

# Using DD Glued


## 1. Creating a DD grid

Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid
from dune.xt.grid import AllDirichletBoundaryInfo
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

macro_grid = make_cube_grid(Dim(d), Cube(), lower_left=omega[0],
                            upper_right=omega[1], num_elements=[4, 4])
# macro_grid.global_refine(1)

macro_boundary_info = AllDirichletBoundaryInfo(macro_grid)

print(f'grid has {macro_grid.size(0)} elements, ' 
      + f'{macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')
```

```python
# the dune way
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF
from dune.xt.functions import IndicatorFunction__2d_to_1x1 as IndicatorFunction
from dune.xt.grid import Dim

from pymor.parameters.functionals import ProjectionParameterFunctional

from itertools import product

def thermal_block_problem_for_dune(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.
    The problem is to solve the elliptic equation ::
      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1
    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block i with value μ_i. ::
           ----------------------------
           |        |        |        |
           |  μ_4   |  μ_5   |  μ_6   |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_1   |  μ_2   |  μ_3   |
           |        |        |        |
           ----------------------------
    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_i is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional('diffusion',
                                             size=num_blocks[0]*num_blocks[1],
                                             index=ix + iy*num_blocks[0],
                                             name=f'diffusion_{ix}_{iy}')

    def diffusion_function_factory(ix, iy):
        dx = 1. / num_blocks[0]
        dy = 1. / num_blocks[1]
        X = [ix * dx, iy * dy]
        Y = [(ix+1) * dx, (iy+1) * dy]
        function = IndicatorFunction([(X, Y, [1.])])
        return function        
    
    rhs= ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='f')

    diffusion_expressions = [diffusion_function_factory(ix, iy)
                             for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))]
    diffusion_functionals = [parameter_functional_factory(ix, iy)
                                for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))]
                            
    return diffusion_expressions, diffusion_functionals, rhs
    
```

```python
diffusion_expr, diffusion_funcs, f = thermal_block_problem_for_dune(blocks)
```

```python
#### PLOT EXPRESSION FUNCTIONS ! 
from dune.xt.functions import GridFunction

for i, expr in enumerate(diffusion_expr):
    expr_grid = GridFunction(macro_grid, expr)
    expr_grid.visualize(macro_grid, f'{i}')
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

# 3. Creating BlockOperators for each expression

```python
from dune.xt.grid import Dim, Walker
from dune.xt.functions import ConstantFunction, ExpressionFunction
from dune.xt.functions import GridFunction

from dune.gdt import DirichletConstraints
```

```python
block_ops = [] 
```

```python
# preparing dirichlet constraints
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

def assemble_subdomain_contribution(grid, space, d, dirichlet_constraints, kappa):
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

def assemble_coupling_contribution(ss, nn, ss_space, nn_space, kappa):
    coupling_grid = dd_grid.coupling_grid(ss, nn)
    
    coupling_sparsity_pattern = make_coupling_sparsity_pattern(ss_space, nn_space, coupling_grid)
        
    coupling_form = BilinearForm(coupling_grid)
    
    # TODO: FIND THE CORRECT NUMBERS HERE ! 
    symmetry_factor = 1.
    weight = 1.
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
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import VectorOperator

for kappa in diffusion_expr:
    ops = np.empty((S, S), dtype=object)
    rhs = np.empty(S, dtype=object)

    for ss in range(S):
        # print(f"macro element: {ss}...")
        local_space = local_spaces[ss]
        local_grid = local_grids[ss]
        dirichlet_constraints = localized_dirichlet_constraints[ss]
        local_op, local_rhs =  assemble_subdomain_contribution(local_grid, local_space, d,
                                                               dirichlet_constraints, kappa)
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
                coupling_ops = assemble_coupling_contribution(ss, nn, local_space,
                                                              neighboring_space, kappa)

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
    
    block_op = BlockOperator(ops)
    block_rhs = VectorOperator(block_op.range.make_array(rhs))
    
    block_ops.append(block_op)
```

```python
coupling_ops[0]
```

```python
from pymor.models.basic import StationaryModel

from pymor.operators.constructions import LincombOperator

block_op = LincombOperator(block_ops, diffusion_funcs)

ipdg = StationaryModel(block_op, block_rhs)

u_ipdg = ipdg.solve(mu)
```

```python
# visualization

from dune.gdt import DiscreteFunction

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

## First reduction globally

```python
from pymor.reductors.coercive import CoerciveRBReductor

us = ipdg.solution_space.empty()

for mu_ in problem.parameter_space.sample_randomly(10):
    us.append(ipdg.solve(mu_))
```

```python
from pymor.algorithms.gram_schmidt import gram_schmidt

us_orth = gram_schmidt(us)

reductor = CoerciveRBReductor(ipdg, us_orth)

rom = reductor.reduce()
```

```python
u_rom_reconstructed = reductor.reconstruct(rom.solve(mu))
```

```python
discrete_functions = []

for ss in range(S):
    u_list_vector_array = u_rom_reconstructed.block(ss)
    u_ss_istl = u_list_vector_array._list[0].real_part.impl
    u_ss = DiscreteFunction(local_spaces[ss], u_ss_istl, name='u_ipdg')
    discrete_functions.append(u_ss)

_ = visualize_discrete_functions_on_dd_grid(discrete_functions, dd_grid)
```

## Reduction locally

```python
from pymor.operators.constructions import ZeroOperator
from pymor.algorithms.projection import project

class EllipticIPDGReductor(CoerciveRBReductor):
    def __init__(self, fom):
        self.S = fom.solution_space.empty().num_blocks
        self.fom = fom
        
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

        # add product blocks
    
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
        for op_blocks in self.ops_blocks:
            ops = np.empty((S, S), dtype=object)
            for ss in range(self.S):
                for nn in range(self.S):
                    local_basis_ss = self.local_bases[ss]
                    local_basis_nn = self.local_bases[nn]
                    ops[ss][nn] = project(op_blocks[ss][nn], local_basis_ss, local_basis_nn)
            projected_ops_blocks.append(BlockOperator(ops))
        for rhs_blocks in self.rhs_blocks:
            rhs = np.empty(S, dtype=object)
            for ss in range(self.S):
                local_basis_ss = self.local_bases[ss]
                rhs_vector = VectorOperator(rhs_blocks.array.block(ss))
                rhs_int = project(rhs_vector, local_basis_ss, None).matrix[:,0]
                rhs[ss] = ops[ss][ss].range.make_array(rhs_int)
                
        projected_operator = LincombOperator(projected_ops_blocks, self.fom.operator.coefficients)
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

localized_reductor = EllipticIPDGReductor(ipdg)
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
discrete_functions = []

for ss in range(S):
    u_list_vector_array = u_rom_loc_reconstructed.block(ss)
    u_ss_istl = u_list_vector_array._list[0].real_part.impl
    u_ss = DiscreteFunction(local_spaces[ss], u_ss_istl, name='u_ipdg')
    discrete_functions.append(u_ss)

_ = visualize_discrete_functions_on_dd_grid(discrete_functions, dd_grid)
```
