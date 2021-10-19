import pickle
import random
import sys
from timeit import default_timer as timer
from typing import Dict

from rich import pretty, traceback

from hapod.cellmodel.wrapper import (
    CellModelSolver,
    DuneCellModel,
    create_parameters,
    solver_statistics,
)
from hapod.mpi import MPIWrapper

traceback.install()
pretty.install()


def solve_and_pickle(mu: Dict[str, float], m: DuneCellModel, chunk_size: int, prefix: str):
    num_chunks, _ = solver_statistics(t_end=m.t_end, dt=m.dt, chunk_size=chunk_size)
    current_values = m.initial_values.copy()
    space = m.solution_space
    t = 0.0
    elapsed = 0.0
    for chunk_index in range(num_chunks):
        filename = (
            f"{prefix}_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}_chunk{chunk_index}.pickle"
        )
        snaps = [space.subspaces[i % 3].empty() for i in range(6)]
        stages = [space.subspaces[i % 3].empty() for i in range(6)]
        residuals = [space.subspaces[i % 3].empty() for i in range(6)]
        # If this is the first time step, add initial values ...
        if chunk_index == 0:
            for k in range(3):
                snaps[k].append(current_values._blocks[k])
        # ... and do one timestep less to ensure that chunk has size chunk_size
        for time_index in range(chunk_size - 1 if chunk_index == 0 else chunk_size):
            t1 = timer()
            current_values, data = m.next_time_step(
                current_values, t, mu=mu, return_stages=True, return_residuals=True
            )
            elapsed += timer() - t1
            t = data["t"]
            # check if we are finished (if time == t_end, m.next_time_step returns None)
            if current_values is None:
                assert chunk_index == num_chunks - 1
                assert time_index != 0
                break
            # store POD input
            for k in range(3):
                snaps[k].append(current_values._blocks[k])
                stages[k].append(data["stages"][k])
                residuals[k].append(data["residuals"][k])
        with open(filename, "wb") as pickle_file:
            pickle.dump(
                {
                    "mu": mu,
                    "snaps": snaps,
                    "stages": stages,
                    "residuals": residuals,
                    "elapsed": elapsed,
                },
                pickle_file,
            )


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
    pol_order = 2
    chunk_size = 10

    ####### choose parameters ####################
    train_params_per_rank = 2
    test_params_per_rank = 1
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

    ########### Choose filename prefix #############
    prefix = "grid{}x{}_tend{}_dt{}_without{}_".format(
        grid_size_x, grid_size_y, t_end, dt, excluded_param
    )
    ########## Solve for chosen parameters #########
    solver = CellModelSolver(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mus[0])
    m = DuneCellModel(solver)
    for mu in mus:
        solve_and_pickle(mu, m, chunk_size, prefix)
        # m.solver.reset();
        print(f"mu: {mu} done")
    for new_mu in new_mus:
        solve_and_pickle(new_mu, m, chunk_size, prefix)
        # m.solver.reset();
        print(f"mu: {new_mu} done")
    mpi.comm_world.Barrier()
