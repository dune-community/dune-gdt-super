import pickle
import random
import sys
from timeit import default_timer as timer

from rich import pretty, traceback

from hapod.cellmodel.wrapper import CellModelSolver, DuneCellModel, create_parameters
from hapod.mpi import MPIWrapper

traceback.install()
pretty.install()

# example call: mpiexec -n 2 python3 cellmodel_solve_with_deim.py single_cell 1e-2 1e-3 30 30 True True
if __name__ == "__main__":
    mpi = MPIWrapper()
    ##### read command line arguments, additional settings #####
    argc = len(sys.argv)
    testcase = "single_cell" if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 30 if argc < 5 else int(sys.argv[4])
    grid_size_y = 30 if argc < 6 else int(sys.argv[5])
    visualize = True if argc < 7 else (False if sys.argv[6] == "False" else True)
    subsampling = True if argc < 8 else (False if sys.argv[7] == "False" else True)
    include_newton_stages = False if argc < 9 else (False if sys.argv[8] == "False" else True)
    pol_order = 2
    chunk_size = 10
    visualize_step = 50
    pod_pfield = True
    pod_ofield = True
    pod_stokes = True
    least_squares_pfield = True
    least_squares_ofield = True
    least_squares_stokes = True

    ####### choose parameters ####################
    train_params_per_rank = 1 if mpi.size_world > 1 else 1
    test_params_per_rank = 1 if mpi.size_world > 1 else 1
    rf = 10  # Factor of largest to smallest training parameter
    random.seed(123)  # create_parameters chooses some parameters randomly in some cases
    excluded_param = "Be"
    mus, new_mus = create_parameters(
        train_params_per_rank,
        test_params_per_rank,
        rf,
        mpi,
        excluded_param,
        None,
        Be0=1.0,
        Ca0=1.0,
        Pa0=1.0,
    )

    # Solve for chosen parameter
    assert len(mus) == 1, "Not yet implemented for more than one parameter per rank"
    solver = CellModelSolver(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mus[0])
    m = DuneCellModel(solver)
    ret, data = m.solve(mu=mus[0], return_stages=include_newton_stages, return_residuals=True)

    ###### Pickle results #########
    prefix = "pickle_grid{}x{}_tend{}_dt{}_{}_without{}_".format(
        grid_size_x,
        grid_size_y,
        t_end,
        dt,
        "snapsandstages" if include_newton_stages else "snaps",
        excluded_param,
    )
    filename_mu = f"{prefix}_Be{mus[0]['Be']}_Ca{mus[0]['Ca']}_Pa{mus[0]['Pa']}"
    filename_new_mu = f"{prefix}_Be{new_mus[0]['Be']}_Ca{new_mus[0]['Ca']}_Pa{new_mus[0]['Pa']}"
    with open(filename_mu, "wb") as pickle_file:
        pickle.dump([mus[0], ret, data], pickle_file)
    del ret, data
    start = timer()
    ret, data = m.solve(mu=new_mus[0], return_stages=include_newton_stages, return_residuals=True)
    data["mean_fom_time"] = (timer() - start) / len(new_mus)
    with open(filename_new_mu, "wb") as pickle_file:
        pickle.dump([new_mus[0], ret, data], pickle_file)

    mpi.comm_world.Barrier()
