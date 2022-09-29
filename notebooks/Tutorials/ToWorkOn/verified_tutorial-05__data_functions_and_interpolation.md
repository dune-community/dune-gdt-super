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

# Tutorial 05 [WIP]: data functions and interpolation

## This is work in progress (WIP), still missing:

* visualization in 1d
* visualization in 3d?

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

## 1: creating functions

We'll work in 2d in this tutorial since scalar functions in 2d can be best visualized within the notebook.
Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[1, 1])

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')
```

Some examples of data functions include analytical functions, such as

$$\begin{align}
f: \mathbb{R}^2 &\to \mathbb{R} && &g: \mathbb{R}^2 &\to \mathbb{R}^{2\times 2}\\
x &\mapsto \alpha&& &x&\mapsto A
\end{align}$$

for a given number $\alpha \in \mathbb{R}$ and matrix $A \in \mathbb{R}^{2 \times 2}$, or

$$\begin{align}
h: \mathbb{R}^2 &\to \mathbb{R}\\
x &\mapsto \text{epx}(x_0\, x_1)
\end{align}$$

or discrete functions $v_h \in V_h$, where $V_h$ is some finite-dimensional discrete function space associated with a grid. Other examples include functions obtained by reading values from a file.

Let's create some of these functions.


## 1.1: using the `ConstantFunction`

For constant functions.

```python
from dune.xt.functions import ConstantFunction

alpha = 0.5
f = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[alpha], name='f')
# note that we have to provide [alpha], since scalar constant functions expect a vector of length 1

A = [[1, 0], [0, 1]]
g = ConstantFunction(dim_domain=Dim(d), dim_range=(Dim(d), Dim(d)), value=A, name='g')
```

## 1.2: using the `ExpressionFunction`

For functions given by an expression, where we have to specify the polynomial order of the expression (or the approximation order for non-polynomial functions).

Note that if the name of the variable is `Foo`, the components `Foo[0]`, ... `Foo[d - 1]` are availabble to be used in the expression.
  
*  We have functions which do not provide a gradient ...

```python
from dune.xt.functions import ExpressionFunction

h = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='exp(x[0]*x[1])', name='h')
```

* ... and functions which provide a gradient, usefull for analytical solutions to compare to and compute $H^1$ errors

```python
h = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='exp(x[0]*x[1])',
                       gradient_expressions=['x[1]*exp(x[0]*x[1])', 'x[0]*exp(x[0]*x[1])'], name='h')
```

## 1.3: discrete functions

Which often result from a discretization scheme, see the *tutorial on continuous Finite Elements for the stationary heat equation*.

```python
from dune.gdt import DiscontinuousLagrangeSpace, DiscreteFunction

V_h = DiscontinuousLagrangeSpace(grid, order=1)
v_h = DiscreteFunction(V_h, name='v_h')

print(v_h.dofs.vector.sup_norm())
```

# 2: visualizing functions

## 2.1: visualizing scalar functions in 2d

We can easily visualize functions mapping from $\mathbb{R}^2 \to \mathbb{R}$. Internally, this is achieved by writing a vtk file to disk and displaying the file using K3D.

```python
from dune.xt.functions import visualize_function

_ = visualize_function(h, grid)
```

**Note**: since functions are always visualized as piecewise linears on each grid element, `dune-grid` supports to write functions on a virtually refined grid, which may be an improvement for higher order data functions. To see this, let us have a look at the rather coarse grid consisting of two simplices:

```python
from dune.xt.grid import visualize_grid

_ = visualize_grid(grid)
```

The two grid elements are clearly visible in the above plot of $h$.
By enabling subsampling, we obtain a much smoother plot, where the underlying virtually refined grid is barely visible.

```python
_ = visualize_function(h, grid, subsampling=True)
```

Subsampling may thus be a means to obtain pretty pictures, but it can also be misleading.


## 2.2: visualizing functions in other dimensions

Functions mapping from and to other dimensions can still be written to disk to be viewed with external viewers such as `paraview`.
Therefore, they need to be wrapped as a `GridFunction`, as explained in the next section.


# 3: The `GridFunction`

No matter the type of data function, we want to be able to use all of these in a discretization scheme without changing the discretization code (e.g., we should be able to pass them all to one of the `discretize_...` functions in `discretize_elliptic_cg.py`).
As explained in *tutorial on continuous Finite Elements for the stationary heat equation*, these functions thus need to be localizable w.r.t. a grid.
While the discrete function $v_h$ is already associated with a grid (namely the `grid` used to create `V_h`), the analytical functions $f$, $g$ and $h$ are not, which poses some problems when writing generic discretization schemes.

Exactly for this purpose, `dune.xt.functions` contains a `GridFunction`, which is a quite powerful means to wrap all kinds of things into a function assocaited with a grid.

For instance, we can wrap any existing function.

```python
from dune.xt.functions import GridFunction

f_grid = GridFunction(grid, f)
g_grid = GridFunction(grid, g)
h_grid = GridFunction(grid, h)
```

In addition, the `GridFunction` directly supports the creation of constant functions.

```python
f_grid = GridFunction(grid, alpha, name='f')
g_grid = GridFunction(grid, A, name='g')
```

Another powerful means to obtain a function is to pass a lambda expression, the first argument of which (`x`) is the point in physical space to evaluate at.

```python
h_grid_lambda = GF(grid,
                   order=10, evaluate_lambda=lambda x, _: [np.exp(x[0]*x[1])],
                   dim_range=Dim(1), name='h')
```

**Note** that using a Python lambda might not yield be the most efficient code, but it is a great way to quickliy implement data functions.


## 3.1: visualizing grid functions in all dimension

To continue *2.2*, we can also write any grid function or discrete function to disk.

```python
f_grid.visualize(grid, 'f') # writes f.vtu
```

```python
!ls -l f.vtu
```

Note that the grid functions `f_grid` is associated with the *type* of the grid `grid`, but not with the actual obejct `grid`.
The function can thus be localized w.r.t. all grids of the same type as `grid`.
For visualization, for instance, we still need to pass an actual `grid` object to visualize w.r.t.

The discrete function, on the other hand, is defined w.r.t. an actual grid object (namely the one used to construct the corresponding discrete function space `V_h`).

```python
v_h.visualize('v_h') # writes v_h.vtu
```

```python
!ls -l v_h.vtu
```

# 4: interpolation

It is often desirable to interpolate data functions in a given discrete function space.


## 4.1: using `dune-gdt`s interpolation

For most discrete function space, the `default_interpolation` will do the right thing, e.g. perform an $L^2$- or Lagrange-Interpolation. There are more specialized interpolations for special cases, e.g. for Raviart-Thomas spaces.

```python
from dune.gdt import default_interpolation, visualize_function # can also visualize discrete functions

h_h = default_interpolation(h_grid, V_h)
_ = visualize_function(h_h, subsampling=False)
```

## 4.2: using vectorized Python code

For Lagrangian discrete function spaces the interpolation can be performed by point evaluation.
We can extract the correct Lagrange-points from the discrete function space and wrap them into a `numpy`-array (without copying them). We can then perform the usual vectorized `numpy` operations and store the result in the DoF-vector of a discrete function (by again wrapping the vector without a copy into a `numpy`-array).

```python
interpolation_points = np.array(V_h.interpolation_points(), copy=False)

h_h_numpy = DiscreteFunction(V_h)
h_h_numpy_vec = np.array(h_h_numpy.dofs.vector, copy=False)
h_h_numpy_vec[:] = np.exp(interpolation_points[..., 0]*interpolation_points[..., 1])

_ = visualize_function(h_h_numpy)
```

# 5: performance considerations

The way a function is constructed may have a large impact on its performance. Let us consider the function $h(x) = exp(x_0\,x_1)$ from above.
We should use a finer grid to obtain some representative timings.

**Note**: by choosing a `Simplex` grid, we obtain an instance of an unstructured `dune-alugrid` grid, which may be runtime efficient, but at the cost of a higher memory footprint.
Choosing a `Cube` grid, on the other hand, would give us an instance of `dune-grid`s structured `YaspGrid` with nearly no memory footprint, at the price of lower runtime performance.

```python
from time import time

t = time()
grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[512, 512])
grid.global_refine(1)
t = time() - t

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices (took {t:.5f}s)')
```

```python
t = time()

V_h = DiscontinuousLagrangeSpace(grid, order=3)

print(f'space has {V_h.num_DoFs} DoFs')
```

## 5.1: interpolation test

One measure of performance is the time it takes to interpolate a data function. Note that the comparison greatly depends on the grid and the polynomial order of `V_h`.

```python
t = time()

h_dune_expression =  GridFunction(grid,
                                  ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='exp(x[0]*x[1])'))

h_dune_expression_h = default_interpolation(h_dune_expression, V_h)

t_dune = time() - t
print(f'interpolating h_dune_expression took {t_dune:.5f}s')
```

```python
t = time()

h_python_lambda = GF(grid,
                     order=10, evaluate_lambda=lambda x, _: [np.exp(x[0]*x[1])],
                     dim_range=Dim(1), name='h')

h_python_lambda_h = default_interpolation(h_python_lambda, V_h)

t_python = time() - t
print(f'interpolating h_python_lambda took {t_python:.5f}s')
```

```python
print(f'using a lambda expression in an interpolation test is {t_python/t_dune:.5f} times slower')
```

## 5.2: discretization test

For a more realistic comparison, we use the `discretize_elliptic_cg_dirichlet_zero` function as explained in the *tutorial on continuous Finite Elements for the stationary heat equation*.

```python
from discretize_elliptic_cg import discretize_elliptic_cg_dirichlet_zero

t = time()

u_h_dune = discretize_elliptic_cg_dirichlet_zero(grid, h_dune_expression, 1)

t_dune = time() - t
print(f'discretizing using h_dune_expression took {t_dune:.5f}s')
```

```python
t = time()

u_h_dune = discretize_elliptic_cg_dirichlet_zero(grid, h_python_lambda, 1)

t_python = time() - t
print(f'discretizing using h_python_lambda took {t_python:.5f}s')
```

```python
print(f'using a lambda expression in a discretization test is {t_python/t_dune:.5f} times slower')
```

This test is much more realistic than the pure interpolation one. Since evaluating the diffusion is only a small part of the overall computation, the performance loss using the Python lambda is much smaller.
Overall, the gain in flexibility outweights the loss in performance (at least for quick tests).
