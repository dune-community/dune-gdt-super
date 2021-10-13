import cProfile
import math
import random
from statistics import mean
import sys
import time
from timeit import default_timer as timer

import numpy as np

from hapod.cellmodel.wrapper import (
    CellModelReductor,
    CellModelSolver,
    DuneCellModel,
)
from hapod.mpi import MPIWrapper
from rich import traceback, print, pretty
import pickle

traceback.install()
pretty.install()


if __name__ == "__main__":
    mpi = MPIWrapper()
    ##### read command line arguments, additional settings #####
    argc = len(sys.argv)
    testcase = "single_cell" if argc < 2 else sys.argv[1]
    t_end = 1e-1 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 90 if argc < 5 else int(sys.argv[4])
    grid_size_y = 90 if argc < 6 else int(sys.argv[5])
    visualize = True if argc < 7 else (False if sys.argv[6] == "False" else True)
    subsampling = True if argc < 8 else (False if sys.argv[7] == "False" else True)
    deim_pfield = True if argc < 9 else (False if sys.argv[8] == "False" else True)
    deim_ofield = True if argc < 10 else (False if sys.argv[9] == "False" else True)
    deim_stokes = True if argc < 11 else (False if sys.argv[10] == "False" else True)
    include_newton_stages = False if argc < 12 else (False if sys.argv[11] == "False" else True)
    pfield_atol = 1e-3 if argc < 13 else float(sys.argv[12])
    ofield_atol = 1e-3 if argc < 14 else float(sys.argv[13])
    stokes_atol = 1e-3 if argc < 15 else float(sys.argv[14])
    pfield_deim_atol = 1e-10 if argc < 16 else float(sys.argv[15])
    ofield_deim_atol = 1e-10 if argc < 17 else float(sys.argv[16])
    stokes_deim_atol = 1e-10 if argc < 18 else float(sys.argv[17])
    pol_order = 2
    chunk_size = 10
    visualize_step = 50
    pod_pfield = True
    pod_ofield = True
    pod_stokes = True
    least_squares_pfield = True
    least_squares_ofield = True
    least_squares_stokes = True
    excluded_param = "Be"
    # default values for parameters
    # Ca0 = 0.1
    # Be0 = 0.3
    Ca0 = 1
    Be0 = 1
    Pa0 = 1.0
    mus = []

    ####### Collect some settings in lists for simpler handling #####
    hapod_tols = [
        pfield_atol,
        ofield_atol,
        stokes_atol,
        pfield_deim_atol,
        ofield_deim_atol,
        stokes_deim_atol,
    ]
    # store fields that should be reduced (0=pfield, 1=ofield, 2=stokes)
    pod_indices = []
    deim_indices = []
    if pod_pfield:
        pod_indices.append(0)
    if deim_pfield:
        deim_indices.append(0)
    if pod_ofield:
        pod_indices.append(1)
    if deim_ofield:
        deim_indices.append(1)
    if pod_stokes:
        pod_indices.append(2)
    if deim_stokes:
        deim_indices.append(2)
    # 0=pfield, 1=ofield, 2=stokes, 3=pfield_deim, 4=ofield_deim, 5=stokes_deim
    indices = pod_indices.copy()
    for index in deim_indices:
        indices.append(index + 3)

    ####### Create training parameters ######
    train_params_per_rank = 1 if mpi.size_world > 1 else 1
    test_params_per_rank = 1 if mpi.size_world > 1 else 1
    num_train_params = train_params_per_rank * mpi.size_world
    num_test_params = test_params_per_rank * mpi.size_world
    # we have two parameters and want to sample both of these parameters with the same number of values
    values_per_parameter_train = int(math.sqrt(num_train_params))
    values_per_parameter_test = int(math.sqrt(num_test_params))

    # Factor of largest to smallest training parameter
    rf = 10
    rf = np.sqrt(rf)
    lower_bound_Ca = Ca0 / rf
    upper_bound_Ca = Ca0 * rf
    lower_bound_Be = Be0 / rf
    upper_bound_Be = Be0 * rf
    lower_bound_Pa = Pa0 / rf
    upper_bound_Pa = Pa0 * rf
    # Compute factors such that mu_i/mu_{i+1} = const and mu_0 = default_value/sqrt(rf), mu_last = default_value*sqrt(rf)
    # factors = np.array([1.])
    factors = np.array(
        [
            (rf ** 2) ** (i / (values_per_parameter_train - 1)) / rf
            for i in range(values_per_parameter_train)
        ]
        if values_per_parameter_train > 1
        else [1.0]
    )
    # Actually create training parameters.
    # Currently, only two parameters vary, the other one is set to the default value
    random.seed(123)
    if excluded_param == "Ca":
        for Be in factors * Be0:
            for Pa in factors * Pa0:
                mus.append({"Pa": Pa, "Be": Be, "Ca": Ca0})
    elif excluded_param == "Be":
        for Ca in factors * Ca0:
            for Pa in factors * Pa0:
                mus.append({"Pa": Pa, "Be": Be0, "Ca": Ca})
    elif excluded_param == "Pa":
        for Ca in factors * Ca0:
            for Be in factors * Be0:
                mus.append({"Pa": Pa0, "Be": Be, "Ca": Ca})
    else:
        raise NotImplementedError(f"Wrong value of excluded_param: {excluded_param}")
    while len(mus) < num_train_params:
        mus.append(
            {
                "Pa": Pa0
                if excluded_param == "Pa"
                else random.uniform(lower_bound_Pa, upper_bound_Pa),
                "Be": Be0
                if excluded_param == "Be"
                else random.uniform(lower_bound_Be, upper_bound_Be),
                "Ca": Ca0
                if excluded_param == "Ca"
                else random.uniform(lower_bound_Ca, upper_bound_Ca),
            }
        )
    ####### Create test parameters ########
    new_mus = []
    for i in range(num_test_params):
        new_mus.append(
            {
                "Pa": Pa0
                if excluded_param == "Pa"
                else random.uniform(lower_bound_Pa, upper_bound_Pa),
                "Be": Be0
                if excluded_param == "Be"
                else random.uniform(lower_bound_Be, upper_bound_Be),
                "Ca": Ca0
                if excluded_param == "Ca"
                else random.uniform(lower_bound_Ca, upper_bound_Ca),
            }
        )

    ###### Start writing output file #########

    filename = "pickle_grid{}x{}_tend{}_dt{}_{}_without{}_".format(
        grid_size_x,
        grid_size_y,
        t_end,
        dt,
        "snapsandstages" if include_newton_stages else "snaps",
        excluded_param,
    )

    if mpi.rank_world == 0:
        with open(filename, "w") as ff:
            ff.write(
                f"{filename}\nTrained with {len(mus)} Parameters: {mus}\n"
                f"Tested with {len(new_mus)} new Parameters: {new_mus}\n"
            )

    ####### Scatter parameters to MPI ranks #######
    # Transform mus and new_mus from plain list to list of lists where the i-th inner list contains all parameters for rank i
    mus = np.reshape(np.array(mus), (mpi.size_world, train_params_per_rank)).tolist()
    new_mus = np.reshape(np.array(new_mus), (mpi.size_world, test_params_per_rank)).tolist()
    mus = mpi.comm_world.scatter(mus, root=0)
    print(f"Mu on rank {mpi.rank_world}: {mus}")
    new_mus = mpi.comm_world.scatter(new_mus, root=0)
    solver = CellModelSolver(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mus[0])
    num_cells = solver.num_cells
    m = DuneCellModel(solver)
    filename_mu = f"{filename}_Be{mus[0]['Be']}_Ca{mus[0]['Ca']}_Pa{mus[0]['Pa']}"
    ret = m.solve(mu=mus[0], return_stages=include_newton_stages, return_residuals=True)
    with open(filename_mu, "wb") as pickle_file:
        pickle.dump([mus[0], ret], pickle_file)
    del ret
    filename_new_mu = f"{filename}_Be{new_mus[0]['Be']}_Ca{new_mus[0]['Ca']}_Pa{new_mus[0]['Pa']}"
    ret = m.solve(mu=new_mus[0], return_stages=include_newton_stages, return_residuals=True)
    with open(filename_new_mu, "wb") as pickle_file:
        pickle.dump([new_mus[0], ret], pickle_file)
    del ret

    mpi.comm_world.Barrier()
