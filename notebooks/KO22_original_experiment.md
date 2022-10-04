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

code from https://github.com/TiKeil/Trust-region-TSRBLOD-code/blob/main/scripts/FINAL_exp_1.py

```python
import numpy as np
from matplotlib import pyplot as plt

from pymor.core.logger import set_log_levels
from pymor.core.defaults import set_defaults
from pymor.core.cache import disable_caching
from pdeopt.tools import print_iterations_and_walltime
set_log_levels({'pymor': 'ERROR',
                'notebook': 'INFO'})
%matplotlib inline

def prepare_kernels():
    set_log_levels({'pymor': 'WARN'})
    # important for the estimator
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-4})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
    disable_caching()

use_pool = False
if use_pool:
    from pymor.parallel.mpi import MPIPool
    pool = MPIPool()
    store_in_tmp = 'tmp'  # <---- adjust this depending on your HPC system 
else:
    from pymor.parallel.dummy import DummyPool
    pool = DummyPool()
    store_in_tmp = False
pool.apply(prepare_kernels)
print_on_ranks = True
```

```python
coarse_elements = 20
n = 200 # 1200
diameter = np.sqrt(2)/n

two_scale_estimator_for_RBLOD = False
save_correctors = False

use_FEM = True
#use_FEM = False
use_fine_mesh = True
#use_fine_mesh = False

# skip_estimator = False
skip_estimator = True

add_error_residual = True
# add_error_residual = False

from pdeopt.problems import large_thermal_block
from pdeopt.discretizer import discretize_quadratic_NCD_pdeopt_stationary_cg
from pdeopt.discretize_gridlod import discretize_gridlod
from pdeopt.discretize_gridlod import discretize_quadratic_pdeopt_with_gridlod

high_conductivity, low_conductivity, min_diffusivity, rhs_value = 4., 1.2, 1., 10.
first_factor, second_factor = 4, 8

print(f'\nVARIABLES: \n'
      f'Coarse elements:        {coarse_elements} x {coarse_elements}\n'
      f'Fine elements:          {n} x {n}\n'
      f'high_c/low_c/min_c:     {high_conductivity}/{low_conductivity}/{min_diffusivity}\n'
      f'rhs/f_1/f_2:            {rhs_value}/{first_factor}/{second_factor}\n')

global_problem, world, local_problem_constructer, f, aFines, f_fine = \
    large_thermal_block(diameter, coarse_elements, blocks=(4, 4), plot=False,
                        return_fine=use_FEM, high_conductivity=high_conductivity,
                        low_conductivity=low_conductivity, rhs_value=rhs_value,
                        first_factor=first_factor, second_factor=second_factor,
                        min_diffusivity=min_diffusivity)

domain_of_interest = None

problem = global_problem

mu_d = global_problem.parameter_space.sample_randomly(1, seed=23)[0]
mu_d_array = mu_d.to_numpy()

# making it harder for the optimizer
for i in [3,4,6,7,8,9,11,14]:
    try:
        mu_d_array[i] = high_conductivity
    except:
        pass
for i in [3,4,5,6]:
    try:
        mu_d_array[i+25] = low_conductivity
    except:
        pass

mu_d = mu_d.parameters.parse(mu_d_array)
norm_mu_d = np.linalg.norm(mu_d_array)
# mu_d = None
mu_for_tikhonov = mu_d

sigma_u = 100
weights = {'sigma_u': sigma_u, 'diffusion': 0.001,
           'low_diffusion': 0.001}

# optional_enrichment = True
```

```python
u_d = None
mu_for_u_d = mu_d
desired_temperature = None
```

```python
# u_d = None
# mu_for_u_d = None
# desired_temperature = 0.1
```

```python
coarse_J = False
if coarse_J is False:
    assert use_fine_mesh
    N_coarse = None
else:
    N_coarse = coarse_elements
```

```python
opt_fom, data, mu_bar = \
    discretize_quadratic_NCD_pdeopt_stationary_cg(problem,
                                                  diameter,
                                                  weights.copy(),
                                                  domain_of_interest=domain_of_interest,
                                                  desired_temperature=desired_temperature,
                                                  mu_for_u_d=mu_for_u_d,
                                                  mu_for_tikhonov=mu_for_tikhonov,
                                                  coarse_functional_grid_size=N_coarse,
                                                  u_d=u_d)

### counting evaluations in opt_fom
from pdeopt.tools import EvaluationCounter, LODEvaluationCounter
counter = EvaluationCounter()

opt_fom = opt_fom.with_(evaluation_counter=counter)
```

```python
'''
    Variables for the optimization algorithms
'''

seed = 1                   # random seed for the starting value
radius = 0.1               # TR radius 
FOC_tolerance = 1e-4       # tau_FOC
sub_tolerance = 1e-8       # tau_sub
safety_tol = 1e-14         # Safeguard, to avoid running the optimizer in machine prevision
max_it = 60                # Maximum number of iteration for the TR algorithm
max_it_sub = 100           # Maximum number of iteration for the TR optimization subproblem
max_it_arm = 50            # Maximum number of iteration for the Armijo rule
init_step_armijo = 0.5     # Initial step for the Armijo rule
armijo_alpha = 1e-4        # kappa_arm
beta = 0.95                # beta_2
epsilon_i = 1e-8           # Treshold for the epsilon active set (Kelley '99)
control_mu = False

# some implementational variables
Qian_Grepl_subproblem = True

reductor_type = 'simple_coercive'
LOD_reductor_type = 'coercive'
if skip_estimator:
    relaxed_reductor_type = 'non_assembled'
    relaxed_LOD_reductor_type = 'non_assembled'
    relaxed_add_error_residual = False
else:
    relaxed_reductor_type = 'simple_coercive'
    relaxed_LOD_reductor_type = 'coercive'
    relaxed_add_error_residual = True

# starting with 
parameter_space = global_problem.parameter_space
mu = parameter_space.sample_randomly(1, seed=seed)[0]

# ### What methods do you want to test ?

optimization_methods = [
    # FOM Method
      'BFGS',
#       'BFGS_LOD',
      'BFGS_IPL',
    # TR-RB
        # NCD-corrected from KMSOV'20
#           'Method_RB', # TR-RB
        # localized BFGS
#           'Method_RBLOD',
#           'Method_TSRBLOD',
            'Method_LRBMS',
    # R TR Methods
      'Method_R_TR'
]
```

```python
parameters = opt_fom.parameters
if mu_for_u_d is not None:
    mu_opt = mu_d
else:
    # adapt this if needed! 
    mu_opt = parameters.parse([1., 2.05833162, 2.41529224, 3.06932311,
                               3.37045877, 2.14031204, 1.2, 1.2])

print('Starting parameter: ', mu.to_numpy())
# J_start_LOD = gridlod_opt_fom.output_functional_hat(mu, pool=pool)
if use_FEM:
    J_start = opt_fom.output_functional_hat(mu)
    print('Starting J FEM: ', J_start)
else:
    J_start = J_start_LOD
    J_opt = J_opt_LOD
# print('Starting J LOD: ', J_start_LOD)
print()
    
mu_opt_as_array = mu_opt.to_numpy()
# J_opt_LOD = gridlod_opt_fom.output_functional_hat(mu_opt, pool=pool)
print('Optimal parameter: ', mu_opt_as_array)
print('Norm mu_d: ', norm_mu_d)
if use_FEM:
    J_opt = opt_fom.output_functional_hat(mu_opt)
    print('Optimal J FEM: ', J_opt)
# print('Optimal J LOD: ', J_opt_LOD)

print('\nParameter space: ', mu_opt.parameters)
```

```python
from pdeopt.tools import compute_errors
from pdeopt.TR import solve_optimization_subproblem_BFGS

counter.reset_counters()
TR_parameters = {'radius': 1.e18, 'sub_tolerance': FOC_tolerance, 
                 'max_iterations_subproblem': 500,
                 'starting_parameter': mu,
                 'epsilon_i': epsilon_i,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'full_order_model': True}

if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    print("\n________________ FOM BFGS________________________\n")
    muoptfom,_,_,_, times_FOM, mus_FOM, Js_FOM, FOC_FOM = \
        solve_optimization_subproblem_BFGS(opt_fom, parameter_space, mu,
                                           TR_parameters, timing=True, FOM=True)
    times_full_FOM, J_error_FOM, mu_error_FOM, FOC = \
        compute_errors(opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array,
                       mus_FOM, Js_FOM, times_FOM, 0, FOC_FOM)
    times_full_FOM = times_full_FOM[1:]
```

```python
if 'BFGS' in optimization_methods or 'All' in optimization_methods:
    print("\n________________ FOM BFGS________________________\n")
    BFGS_dict = counter.print_result(True)
    # print_RB_result(BFGS_dict)
    print_iterations_and_walltime(len(times_full_FOM), times_full_FOM[-1])
    print('mu_error: ', mu_error_FOM[-1]) 
counter.reset_counters()
```

```python
'''
    ROM OPTIMIZATION ALGORITHMS
'''

import time
from pdeopt.model import build_initial_basis
from pdeopt.reductor import QuadraticPdeoptStationaryCoerciveReductor
from pdeopt.TR import TR_algorithm
from pdeopt.relaxed_TR import Relaxed_TR_algorithm
from pymor.parameters.functionals import MinThetaParameterFunctional

set_defaults({'pymor.operators.constructions.induced_norm.raise_negative': False})
set_defaults({'pymor.operators.constructions.induced_norm.tol': 1e-20})

ce = MinThetaParameterFunctional(opt_fom.primal_model.operator.coefficients, mu_bar)

# ## NCD corrected BFGS Method (KMSOV'20)
counter.reset_counters()

tic = time.time()
params = [mu]

if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) or \
    'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)

    RBbasis, dual_RBbasis = build_initial_basis(
        opt_fom, params, build_sensitivities=False)
    
    pdeopt_reductor = \
        QuadraticPdeoptStationaryCoerciveReductor(opt_fom, 
                                                  RBbasis, dual_RBbasis, 
                                                  opt_product=opt_fom.opt_product,
                                                  coercivity_estimator=ce,
                                                  reductor_type=relaxed_reductor_type,
                                                  mu_bar=mu_bar)

    opt_rom = pdeopt_reductor.reduce()

    tictoc = time.time() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': 0.1, 'FOC_tolerance': FOC_tolerance, 
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub, 
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo, 
                 'armijo_alpha': armijo_alpha, 
                 'epsilon_i': epsilon_i, 
                 'control_mu': control_mu,
                 'starting_parameter': mu, 
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True, 
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_ntr8, times_ntr8, Js_ntr8, FOC_ntr8, data_ntr8 = \
        Relaxed_TR_algorithm(opt_rom, pdeopt_reductor, parameter_space,
                             TR_parameters, extension_params, skip_estimator=skip_estimator)
    
    times_full_ntr8_actual, J_error_ntr8_actual, mu_error_ntr8_actual, FOC_ntr8_actual = \
        compute_errors(opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array,
                       mus_ntr8, Js_ntr8, times_ntr8, tictoc, FOC_ntr8, pool=pool)
```

```python
if ('Method_R_TR' in optimization_methods and 'Method_RB' in optimization_methods) \
    or 'All' in optimization_methods:
    print("\n_________________Relaxed TR NCD BFGS_____________________\n")
    R_TRNCDRB_dict = counter.print_result(True)
    # print_RB_result(R_TRNCDRB_dict)
    print_iterations_and_walltime(len(times_full_ntr8_actual), times_full_ntr8_actual[-1])
    print('mu_error: ', mu_error_ntr8_actual[-1])
    subproblem_time = data_ntr8['total_subproblem_time']
    print(f'further timings:\n subproblem:  {subproblem_time:.3f}')

counter.reset_counters()
```

```python
tic = time.time()
params = [mu]

# TODO: FIX THE VERSION OF PYMOR HERE !!! 
if 0 and 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________TR NCD BFGS_____________________\n")
    # make sure to use the correct config
    opt_fom = opt_fom.with_(use_corrected_functional=True)
    opt_fom = opt_fom.with_(adjoint_approach=True)

    RBbasis, dual_RBbasis = build_initial_basis(
        opt_fom, params, build_sensitivities=False)
    
    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveReductor(opt_fom, 
                                                RBbasis, dual_RBbasis, 
                                                opt_product=opt_fom.opt_product,
                                                coercivity_estimator=ce,
                                                reductor_type=reductor_type, mu_bar=mu_bar)

    opt_rom = pdeopt_reductor.reduce()
    
    tictoc = time.time() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': radius, 'FOC_tolerance': FOC_tolerance, 
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub, 
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo, 
                 'armijo_alpha': armijo_alpha, 
                 'epsilon_i': epsilon_i, 
                 'control_mu': control_mu,
                 'starting_parameter': mu, 
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True, 
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_8, times_8, Js_8, FOC_8, data_8 = TR_algorithm(opt_rom, pdeopt_reductor,
                                                       parameter_space, TR_parameters,
                                                       extension_params)
    
    times_full_8_actual, J_error_8_actual, mu_error_8_actual, FOC_8_actual = \
        compute_errors(opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array,
                       mus_8, Js_8, times_8, tictoc, FOC_8, pool=pool)


if 0 and 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    plt.semilogy(times_full_8_actual, FOC_8_actual)
```

```python
if 0 and 'Method_RB' in optimization_methods or 'All' in optimization_methods:
    print("\n_________________TR NCD BFGS_____________________\n")
    TRNCDRB_dict = counter.print_result(True)
    # print_RB_result(TRNCDRB_dict)
    print_iterations_and_walltime(len(times_full_8_actual), times_full_8_actual[-1])
    print('mu_error: ', mu_error_8_actual[-1])
    subproblem_time = data_8['total_subproblem_time']
    print(f'further timings:\n subproblem:  {subproblem_time:.3f}')

counter.reset_counters()
```

# LRBMS

```python
from pdeopt_discretizer import discretize_quadratic_pdeopt_with_iplrb
```

```python
N = 5.
macro_diameter = np.sqrt(2)/N
# automatically detect how many refinements are needed
refinements = int(np.log2(n/N))

ipl_opt_fom, data, mu_bar = \
    discretize_quadratic_pdeopt_with_iplrb(global_problem,
                                           macro_diameter,
                                           refinements=refinements,
                                           symmetry_factor=1.,
                                           weight_parameter=None,
                                           penalty_parameter=16.,
                                           weights=weights.copy(),
                                           domain_of_interest=domain_of_interest,
                                           desired_temperature=desired_temperature,
                                           mu_for_u_d=mu_for_u_d,
                                           mu_for_tikhonov=mu_for_tikhonov,
                                           pool=pool,
                                           counter=None,
                                           store_in_tmp=store_in_tmp,
                                           coarse_J=coarse_J,
                                           use_fine_mesh=use_fine_mesh,
                                           aFine_constructor=local_problem_constructer,
                                           u_d=u_d,
                                           print_on_ranks=print_on_ranks)
```

```python
macro_grid = data['macro_grid']
dd_grid = data['dd_grid']
local_spaces = data['local_spaces']
```

```python
J_opt_ipl = ipl_opt_fom.output_functional_hat(mu_opt)
J_start_ipl = ipl_opt_fom.output_functional_hat(mu)
print('Optimal J IPL: ', J_opt_ipl)
print('Optimal J FEM: ', J_opt)
print('Starting J IPL: ', J_start_ipl)
print('Starting J FEM: ', J_start)
```

```python
TR_parameters = {'radius': 1.e18, 'sub_tolerance': FOC_tolerance, 
                 'max_iterations_subproblem': 500,
                 'starting_parameter': mu,
                 'epsilon_i': epsilon_i,
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo,
                 'armijo_alpha': armijo_alpha,
                 'full_order_model': True}
counter.reset_counters()

if 'BFGS_IPL' in optimization_methods or 'All' in optimization_methods:
    print("\n________________IPL FOM BFGS________________________\n")
    muoptfom,_,_,_, times_ipl_FOM, mus_ipl_FOM, Js_ipl_FOM, FOC_ipl_FOM = \
            solve_optimization_subproblem_BFGS(ipl_opt_fom, parameter_space, mu,
                                               TR_parameters, timing=True, FOM=True)
    times_full_ipl_FOM, J_error_ipl_FOM, mu_error_ipl_FOM, FOC = \
            compute_errors(ipl_opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array, 
                           mus_ipl_FOM, Js_ipl_FOM, times_ipl_FOM, 0, FOC_ipl_FOM)
    times_full_ipl_FOM = times_full_ipl_FOM[1:]
```

```python
if 'BFGS_IPL' in optimization_methods or 'All' in optimization_methods:
    print("\n________________IPL FOM BFGS________________________\n")
    IPL_BFGS_dict = counter.print_result(True)
    # print_RB_result(BFGS_dict)
    print_iterations_and_walltime(len(times_full_ipl_FOM), times_full_ipl_FOM[-1])
    print('mu_error: ', mu_error_ipl_FOM[-1]) 
counter.reset_counters()
```

```python
from pdeopt_reductor import QuadraticPdeoptStationaryCoerciveLRBMSReductor

if ('Method_R_TR' in optimization_methods and 'Method_LRBMS' in optimization_methods) \
        or 'All' in optimization_methods:
    print("\n_________________Relaxed TR LRBMS BFGS_____________________\n")
    
    # make sure to use the correct config
    iplrb_opt_fom = ipl_opt_fom.with_(use_corrected_functional=False,
                                      adjoint_approach=False)
    
    print('constructing reductor ...')
    pdeopt_reductor = QuadraticPdeoptStationaryCoerciveLRBMSReductor(
        ipl_opt_fom, f, dd_grid=dd_grid, opt_product=iplrb_opt_fom.opt_product,
        coercivity_estimator=ce, reductor_type='non_assembled',
        mu_bar=mu_bar, parameter_space=parameter_space,
        pool=pool, optional_enrichment=False, store_in_tmp=store_in_tmp,
        use_fine_mesh=use_fine_mesh,
        print_on_ranks=print_on_ranks
    )
    print('enriching ...')
    
    pdeopt_reductor.add_global_solutions(mu)

    print('construct ROM ...')
    opt_rom = pdeopt_reductor.reduce()

    tictoc = time.time() - tic

    TR_parameters = {'Qian-Grepl_subproblem': Qian_Grepl_subproblem, 'beta': beta,
                 'safety_tolerance': safety_tol,
                 'radius': 0.1, 'FOC_tolerance': FOC_tolerance, 
                 'sub_tolerance': sub_tolerance,
                 'max_iterations': max_it, 'max_iterations_subproblem': max_it_sub, 
                 'max_iterations_armijo': max_it_arm,
                 'initial_step_armijo': init_step_armijo, 
                 'armijo_alpha': armijo_alpha, 
                 'epsilon_i': epsilon_i, 
                 'control_mu': control_mu,
                 'starting_parameter': mu, 
                 'opt_method': 'BFGSMethod'}

    extension_params = {'Enlarge_radius': True, 'timings': True, 
                        'opt_fom': opt_fom, 'return_data_dict': True}

    mus_rlrb, times_rlrb, Js_rlrb, FOC_rlrb, data_rlrb = \
        Relaxed_TR_algorithm(opt_rom, pdeopt_reductor, parameter_space, TR_parameters,
                             extension_params, skip_estimator=skip_estimator)
    
    times_full_rlrb_actual, J_error_rlrb_actual, mu_error_rlrb_actual, FOC_rlrb_actual = \
        compute_errors(ipl_opt_fom, parameter_space, J_start, J_opt, mu, mu_opt_as_array,
                       mus_rlrb, Js_rlrb, times_rlrb, tictoc, FOC_rlrb, pool=pool)
```

```python
if ('Method_R_TR' in optimization_methods and 'Method_LRBMS' in optimization_methods) \
    or 'All' in optimization_methods:
    print("\n_________________TR LRBMS BFGS_____________________\n")
    TRLRBMS_dict = counter.print_result(True)
    # print_RB_result(TRLRBMS_dict)
    print_iterations_and_walltime(len(times_full_rlrb_actual), times_full_rlrb_actual[-1])
    print('mu_error: ', mu_error_rlrb_actual[-1])
    subproblem_time = data_rlrb['total_subproblem_time']
    print(f'further timings:\n subproblem:  {subproblem_time:.3f}')

counter.reset_counters()
```
