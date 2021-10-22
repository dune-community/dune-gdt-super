# import cProfile
import pickle
import random
import resource
from statistics import mean
import sys
from timeit import default_timer as timer
from typing import Any, Dict, Union

from hapod.cellmodel.wrapper import (
    BinaryTreeHapodResults,
    CellModelOfieldProductOperator,
    CellModelPfieldProductOperator,
    CellModelReductor,
    CellModelSolver,
    CellModelStokesProductOperator,
    DuneCellModel,
    calculate_cellmodel_errors,
    create_parameters,
    final_hapod_in_binary_tree_hapod,
    pod_on_node_in_binary_tree_hapod,
    pods_on_processor_cores_in_binary_tree_hapod,
    solver_statistics,
)
from hapod.hapod import binary_tree_depth
from hapod.mpi import MPIWrapper
import numpy as np
from rich import pretty, print, traceback

traceback.install()
pretty.install()


def binary_tree_hapod(
    cellmodel: DuneCellModel,
    prefix: str,
    t_end: float,
    dt: float,
    mus: "list[Dict[str, float]]",
    chunk_size: int,
    mpi: MPIWrapper,
    tolerances: "list[float]",
    indices: "list[int]",
    include_newton_stages: bool = False,
    omega: float = 0.95,
    incremental_gramian: bool = False,
    orth_tol: float = np.inf,
    # orth_tol=1e-10,
    final_orth_tol: float = 1e-10,
    logfile: Union[None, str] = None,
    products: "list[Any]" = [None] * 6,
):
    assert len(tolerances) == 6

    # setup timings
    timings = {}
    for k in indices:
        timings[f"POD{k}"] = 0.0

    # calculate rooted tree depth
    num_chunks, _ = solver_statistics(t_end=t_end, dt=dt, chunk_size=chunk_size)
    node_binary_tree_depth = binary_tree_depth(
        mpi.comm_rank_group[0]
    )  # assumes same tree for all fields
    node_binary_tree_depth = mpi.comm_proc.bcast(node_binary_tree_depth, root=0)
    tree_depth = num_chunks + node_binary_tree_depth

    # create classes that store HAPOD results and parameters for easier handling
    results = [BinaryTreeHapodResults(tree_depth, tol, omega) for tol in tolerances]
    # add some more properties to the results classes
    for r in results:
        r.orth_tol = orth_tol
        r.final_orth_tol = final_orth_tol
        r.incremental_gramian = incremental_gramian

    # walk over time chunks
    for chunk_index in range(num_chunks):
        new_vecs = [cellmodel.solution_space.subspaces[i % 3].empty() for i in range(6)]
        for mu in mus:
            filename = f"{prefix}_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}_chunk{chunk_index}.pickle"
            with open(filename, "rb") as pickle_file:
                data = pickle.load(pickle_file)
            for k in indices:
                if k < 3:
                    # this is a POD index
                    new_vecs[k].append(data["snaps"][k])
                    if include_newton_stages:
                        new_vecs[k].append(data["stages"][k])
                else:
                    # this is a DEIM index
                    new_vecs[k].append(data["residuals"][k - 3])

        # calculate POD of timestep vectors on each core
        for k in indices:
            t1 = timer()
            root_rank = k % mpi.size_proc
            pods_on_processor_cores_in_binary_tree_hapod(
                results[k], new_vecs[k], mpi, root=root_rank, product=products[k]
            )
            timings[f"POD{k}"] += timer() - t1
        for k in indices:
            t1 = timer()
            # perform PODs in parallel for each field
            root_rank = k % mpi.size_proc
            if mpi.rank_proc == root_rank:
                # perform pod on rank root_rank with gathered modes and modes from the last chunk
                pod_on_node_in_binary_tree_hapod(
                    results[k], chunk_index, num_chunks, mpi, root=root_rank, product=products[k]
                )
            timings[f"POD{k}"] += timer() - t1

    # Finally, perform a HAPOD over a binary tree of nodes
    for k in indices:
        root_rank = k % mpi.size_proc
        r = results[k]
        if mpi.rank_proc == root_rank:
            t1 = timer()
            final_hapod_in_binary_tree_hapod(r, mpi, root=root_rank, product=products[k])
            timings[f"POD{k}"] += timer() - t1

        # calculate max number of local modes
        # The 'or [0]' is only here to silence pyright which otherwise complains below that we cannot apply max to None
        r.max_vectors_before_pod = mpi.comm_world.gather(r.max_vectors_before_pod, root=0) or [0]
        r.max_local_modes = mpi.comm_world.gather(r.max_local_modes, root=0) or [0]
        r.num_modes = mpi.comm_world.gather(len(r.modes) if r.modes is not None else 0, root=0) or [
            0
        ]
        r.total_num_snapshots = mpi.comm_world.gather(r.total_num_snapshots, root=0) or [0]
        gathered_timings = mpi.comm_world.gather(timings, root=0) or [0]
        if mpi.rank_world == 0:
            r.max_vectors_before_pod = max(r.max_vectors_before_pod)
            r.max_local_modes = max(r.max_local_modes)
            r.num_modes = max(r.num_modes)
            r.total_num_snapshots = max(r.total_num_snapshots)
            r.timings = {}
            for key in timings.keys():
                r.timings[key] = max([timing[key] for timing in gathered_timings])

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        with open(logfile, "a") as ff:
            for k in indices:
                r = results[k]
                ff.write(f"Hapod for index {k}\n")
                ff.write(
                    f"The HAPOD resulted in {r.num_modes} final modes taken from a total of {r.total_num_snapshots} snapshots!\n"
                )
                ff.write(f"The maximal number of local modes was {r.max_local_modes}\n")
                ff.write(
                    f"The maximal number of input vectors to a local POD was: {r.max_vectors_before_pod}\n"
                )
                ff.write("PODs took {} s.\n".format(r.timings[f"POD{k}"]))
            ff.write(
                f"The maximum amount of memory used on rank 0 was: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 ** 2} GB\n"
            )
    mpi.comm_world.Barrier()
    return results, num_chunks


# exemplary call: mpiexec -n 2 python3 cellmodel_hapod_with_pickled_data.py single_cell 1e-2 1e-3 30 30 True True False False False False 1e-3 1e-3 1e-3 1e-10 1e-10 1e-10
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
    # use_L2_product = False
    use_L2_product = True
    train_params_per_rank = 2
    test_params_per_rank = 1

    ###### Choose filename #########
    filename = "results_pickled_{}procs_{}_grid{}x{}_tend{}_dt{}_{}_without{}_{}tppr_pfield{}_ofield{}_stokes{}.txt".format(
        mpi.size_world,
        "L2product" if use_L2_product else "noproduct",
        grid_size_x,
        grid_size_y,
        t_end,
        dt,
        "snapsandstages" if include_newton_stages else "snaps",
        excluded_param,
        train_params_per_rank,
        (f"pod{pfield_atol:.0e}" if pod_pfield else "pod0")
        + (f"deim{pfield_deim_atol:.0e}" if deim_pfield else "deim0"),
        (f"pod{ofield_atol:.0e}" if pod_ofield else "pod0")
        + (f"deim{ofield_deim_atol:.0e}" if deim_ofield else "deim0"),
        (f"pod{stokes_atol:.0e}" if pod_stokes else "pod0")
        + (f"deim{stokes_deim_atol:.0e}" if deim_stokes else "deim0"),
    )

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

    ####### choose parameters ####################
    rf = 10  # Factor of largest to smallest training parameter
    random.seed(123)  # create_parameters choose some parameters randomly in some cases
    mus, new_mus = create_parameters(
        train_params_per_rank,
        test_params_per_rank,
        rf,
        mpi,
        excluded_param,
        filename,
        Be0=1.0,
        Ca0=1.0,
        Pa0=1.0,
    )

    ######  same filenames as in cellmodel_write_data.py     ##########
    prefix = "grid{}x{}_tend{}_dt{}_without{}_".format(
        grid_size_x, grid_size_y, t_end, dt, excluded_param
    )
    solver = CellModelSolver(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mus[0])
    if use_L2_product:
        products = [
            CellModelPfieldProductOperator(solver),
            CellModelOfieldProductOperator(solver),
            CellModelStokesProductOperator(solver),
        ] * 2
    else:
        products = [None] * 6
    m = DuneCellModel(solver,
        products={"pfield": products[0], "ofield": products[1], "stokes": products[2]})
    results, num_chunks = binary_tree_hapod(
        cellmodel=m,
        prefix=prefix,
        t_end=t_end,
        dt=dt,
        mus=mus,
        chunk_size=chunk_size,
        mpi=mpi,
        tolerances=hapod_tols,
        indices=indices,
        include_newton_stages=include_newton_stages,
        omega=0.95,
        logfile=filename,
        products=products,
    )
    for k in indices:
        r = results[k]
        r.modes, r.win = mpi.shared_memory_bcast_modes(
            r.modes, returnlistvectorarray=True, proc_rank=k % mpi.size_proc
        )

    pfield_basis = results[0].modes if pod_pfield else None
    ofield_basis = results[1].modes if pod_ofield else None
    stokes_basis = results[2].modes if pod_stokes else None
    pfield_deim_basis = results[3].modes if deim_pfield else None
    ofield_deim_basis = results[4].modes if deim_ofield else None
    stokes_deim_basis = results[5].modes if deim_stokes else None

    reduced_prefix = "without{}_{}_pfield_{}_ofield_{}_stokes_{}".format(
        excluded_param,
        "snapsandstages" if include_newton_stages else "snaps",
        (f"pod{pfield_atol:.0e}" if pod_pfield else "pod0")
        + (f"deim{pfield_deim_atol:.0e}" if deim_pfield else "deim0"),
        (f"pod{ofield_atol:.0e}" if pod_ofield else "pod0")
        + (f"deim{ofield_deim_atol:.0e}" if deim_ofield else "deim0"),
        (f"pod{stokes_atol:.0e}" if pod_stokes else "pod0")
        + (f"deim{stokes_deim_atol:.0e}" if deim_stokes else "deim0"),
    )

    reductor = CellModelReductor(
        m,
        pfield_basis,
        ofield_basis,
        stokes_basis,
        least_squares_pfield=least_squares_pfield,
        least_squares_ofield=least_squares_ofield,
        least_squares_stokes=least_squares_stokes,
        pfield_deim_basis=pfield_deim_basis,
        ofield_deim_basis=ofield_deim_basis,
        stokes_deim_basis=stokes_deim_basis,
        check_orthonormality=True,
        check_tol=1e-10,
        products={"pfield": products[0], "ofield": products[1], "stokes": products[2]},
    )
    del solver, m
    rom = reductor.reduce()

    ################## solve reduced model for trained parameters ####################
    us = []
    for mu in mus:
        u, _ = rom.solve(mu, return_stages=False)
        us.append(u)
        # U_rom = reductor.reconstruct(u)
        # m.visualize(U_rom, prefix=f"noprod_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}", subsampling=subsampling, every_nth=visualize_step)

    ########## Compute errors for trained parameters #################
    sys.stdout.flush()
    mpi.comm_world.Barrier()
    calculate_cellmodel_errors(
        modes=[pfield_basis, ofield_basis, stokes_basis],
        deim_modes=[pfield_deim_basis, ofield_deim_basis, stokes_deim_basis],
        testcase=testcase,
        t_end=t_end,
        dt=dt,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        pol_order=pol_order,
        mus=mus,
        reduced_us=us,
        reductor=reductor,
        mpi_wrapper=mpi,
        logfile_name=filename,
        products=products,
        pickled_data_available=True,
        num_chunks=num_chunks,
        pickle_prefix=prefix,
    )

    ################## test new parameters #######################
    start = timer()
    # cProfile.run(
    #     "u_new_mu = rom.solve(new_mus[0], return_stages=False)", f"rom{mpi.rank_world}.cprof"
    # )
    us_new_mu = []
    for new_mu in new_mus:
        u, _ = rom.solve(new_mu, return_stages=False)
        us_new_mu.append(u)
    mean_rom_time = (timer() - start) / len(new_mus)

    ############### Compute errors for new parameters ##################
    mpi.comm_world.Barrier()
    calculate_cellmodel_errors(
        modes=[pfield_basis, ofield_basis, stokes_basis],
        deim_modes=[pfield_deim_basis, ofield_deim_basis, stokes_deim_basis],
        testcase=testcase,
        t_end=t_end,
        dt=dt,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        pol_order=pol_order,
        mus=new_mus,
        reduced_us=us_new_mu,
        reductor=reductor,
        mpi_wrapper=mpi,
        logfile_name=filename,
        prefix="new ",
        products=products,
        pickled_data_available=True,
        num_chunks=num_chunks,
        pickle_prefix=prefix,
        rom_time=mean_rom_time,
    )
    sys.stdout.flush()
    mpi.comm_world.Barrier()
