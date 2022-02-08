import os
import numpy as np
import pickle
import random
import sys
from timeit import default_timer as timer
from typing import Dict

from hapod.cellmodel.wrapper import (
    CellModelOfieldL2ProductOperator,
    CellModelPfieldL2ProductOperator,
    CellModelOfieldH1ProductOperator,
    CellModelPfieldH1ProductOperator,
    CellModelSolver,
    CellModelStokesL2ProductOperator,
    DuneCellModel,
    create_parameters,
    solver_statistics,
)
from hapod.mpi import MPIWrapper
from rich import pretty, traceback

traceback.install()
pretty.install()

# If true, directly reads the pickled data back in and compares it to the written data
check_pickling = False

def solve_and_pickle(mu: Dict[str, float], m: DuneCellModel, chunk_size: int, pickle_prefix: str):
    num_chunks, _ = solver_statistics(t_end=m.t_end, dt=m.dt, chunk_size=chunk_size)
    current_values = m.initial_values.copy()
    space = m.solution_space
    t = 0.0
    elapsed = 0.0
    for chunk_index in range(num_chunks):
        filename = (
            f"{pickle_prefix}_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}_chunk{chunk_index}.pickle"
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
        if check_pickling:
            with open(filename, "rb") as pickle_file:
                data = pickle.load(pickle_file)
                # print(str(data["snaps"]), "\nvs.", str(snaps), flush=True)
                np.set_printoptions(precision=15)
                snaps2 = data["snaps"]
                stages2 = data["stages"]
                residuals2 = data["residuals"]
                for i in range(len(snaps)):
                    for j in range(len(snaps[i])):
                        for z in range(snaps[i]._list[j].dim):
                            if snaps2[i]._list[j].to_numpy()[z] != snaps[i]._list[j].to_numpy()[z]:
                                print(f"{snaps2[i][j][z]} vs. {snaps[i][j][z]}", flush=True)
                                raise NotImplementedError
                    for j in range(len(stages[i])):
                        for z in range(stages[i]._list[j].dim):
                            if stages2[i]._list[j].to_numpy()[z] != stages[i]._list[j].to_numpy()[z]:
                                print(f"{stages2[i][j][z]} vs. {stages[i][j][z]}", flush=True)
                                raise NotImplementedError
                    for j in range(len(residuals[i])):
                        for z in range(residuals[i]._list[j].dim):
                            if residuals2[i]._list[j].to_numpy()[z] != residuals[i]._list[j].to_numpy()[z]:
                                print(f"{residuals2[i][j][z]} vs. {residuals[i][j][z]}", flush=True)
                                raise NotImplementedError


# exemplary call: mpiexec -n 2 python3 cellmodel_write_data.py single_cell 1e-2 1e-3 30 30
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
    # product_type = "L2"
    product_type = "H1"
    train_params_per_rank = 1
    test_params_per_rank = 1
    random.seed(123)  # create_parameters chooses some parameters randomly in some cases

    ####### choose parameters ####################
    rf = 5  # Factor of largest to smallest training parameter
    excluded_params = ("Be", "Ca")
    mus, new_mus = create_parameters(
        train_params_per_rank,
        test_params_per_rank,
        rf,
        mpi,
        excluded_params,
        None,
        Be0=1.0,
        Ca0=1.0,
        Pa0=1.0,
    )

    ########### Choose filename prefix #############
    pickle_dir = "pickle_files"
    if mpi.rank_world == 0:
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)
    pickle_dir = "pickle_files"
    pickle_prefix = f"{testcase}_grid{grid_size_x}x{grid_size_y}_tend{t_end}_dt{dt}"
    pickle_prefix = os.path.join(
        pickle_dir,
        pickle_prefix
    )
    ########## Create products #####################
    solver = CellModelSolver(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mus[0])
    if product_type == "L2":
        products = [
            CellModelPfieldL2ProductOperator(solver),
            CellModelOfieldL2ProductOperator(solver),
            CellModelStokesL2ProductOperator(solver),
        ] * 2
    elif product_type == "H1":
        products = [
            CellModelPfieldH1ProductOperator(solver),
            CellModelOfieldL2ProductOperator(solver),
            CellModelStokesL2ProductOperator(solver),
        ] * 2
    else:
        products = [None] * 6

    ########## Solve for chosen parameters #########
    m = DuneCellModel(
        solver, products={"pfield": products[0], "ofield": products[1], "stokes": products[2]}
    )
    for mu in mus:
        solve_and_pickle(mu, m, chunk_size, pickle_prefix)
        # m.solver.reset();
        print(f"mu: {mu} done")
    for new_mu in new_mus:
        solve_and_pickle(new_mu, m, chunk_size, pickle_prefix)
        # m.solver.reset();
        print(f"mu: {new_mu} done")
    mpi.comm_world.Barrier()
