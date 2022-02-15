# import cProfile
import os
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
    CellModelReductor,
    CellModelSolver,
    CellModelStokesL2ProductOperator,
    DuneCellModel,
    binary_tree_hapod,
    binary_tree_hapod,
    calculate_cellmodel_errors,
    create_parameters,
    solver_statistics,
)
from hapod.mpi import MPIWrapper
import numpy as np
from rich import pretty, traceback

traceback.install()
pretty.install()

# If this is set to true, we check the computed solutions against the pickled solutions (which have to be available in that case)
# Should usually be False, enable only for debug purposes (to ensure that we get the same result with and without pickling)
check_against_pickled = False
# There seem to be some errors in the order of 1e-13.
# Not sure where these come from, pickling and unpickling the data does not seem to introduce any errors (see cellmodel_write_data.py)
check_atol = 2e-12
check_rtol = 0


class SolverChunkGenerator:
    def __init__(
        self,
        pickle_prefix: str,
        cellmodel: DuneCellModel,
        t_end: float,
        dt: float,
        chunk_size: int,
        mus: "list[Dict[str, float]]",
        include_newton_stages: bool,
        indices: "list[int]",
        products: "list[Any]",
        normalize_residuals: bool,
    ):
        self.pickle_prefix = pickle_prefix
        self.cellmodel = cellmodel
        self.mus = mus
        self.include_newton_stages = include_newton_stages
        self.indices = indices
        self.current_chunk_index = -1
        self.chunk_size = chunk_size
        self.num_chunks, _ = solver_statistics(t_end=t_end, dt=dt, chunk_size=chunk_size)
        self.products = products
        self.normalize_residuals = normalize_residuals

    def __iter__(self):
        # walk over time chunks
        # currently, all parameters mu have to use the same timestep length in all time steps
        old_t = 0.0
        t = 0.0
        current_values = [self.cellmodel.initial_values.copy() for _ in range(len(self.mus))]
        for chunk_index in range(self.num_chunks):
            self.current_chunk_index = chunk_index
            new_vecs = [self.cellmodel.solution_space.subspaces[i % 3].empty() for i in range(6)]
            if check_against_pickled:
                new_vecs2 = [self.cellmodel.solution_space.subspaces[i % 3].empty() for i in range(6)]
            # walk over parameters
            for p, mu in enumerate(self.mus):
                t = old_t
                # If this is the first time step, add initial values ...
                if chunk_index == 0:
                    for k in self.indices:
                        if k < 3:  # POD indices
                            new_vecs[k].append(current_values[p]._blocks[k])
                # ... and do one timestep less to ensure that chunk has size chunk_size
                for time_index in range(self.chunk_size - 1 if chunk_index == 0 else self.chunk_size):
                    # t1 = timer()
                    current_values[p], data = self.cellmodel.next_time_step(
                        current_values[p], t, mu=mu, return_stages=self.include_newton_stages, return_residuals=True
                    )
                    # timings["data"] += timer() - t1
                    t = data["t"]
                    # check if we are finished (if time == t_end, cellmodel.next_time_step returns None)
                    if current_values[p] is None:
                        assert chunk_index == self.num_chunks - 1
                        assert time_index != 0
                        break
                    # store POD input
                    for k in self.indices:
                        if k < 3:
                            # this is a POD index
                            new_vecs[k].append(current_values[p]._blocks[k])
                            if self.include_newton_stages:
                                new_vecs[k].append(data["stages"][k])
                        else:
                            # this is a DEIM index
                            residuals = data["residuals"][k - 3]
                            if self.normalize_residuals:
                                residuals = residuals * np.array(
                                    [1 / norm if norm > 0 else 1 for norm in residuals.norm(product=self.products[k])]
                                )
                            new_vecs[k].append(residuals)

                if check_against_pickled:
                    filename = f"{self.pickle_prefix}_Be{mu['Be']}_Ca{mu['Ca']}_Pa{mu['Pa']}_chunk{chunk_index}.pickle"
                    with open(filename, "rb") as pickle_file:
                        data = pickle.load(pickle_file)
                    for k in self.indices:
                        if k < 3:
                            # this is a POD index
                            new_vecs2[k].append(data["snaps"][k])
                            if self.include_newton_stages:
                                new_vecs2[k].append(data["stages"][k])
                        else:
                            # this is a DEIM index
                            residuals = data["residuals"][k - 3]
                            if self.normalize_residuals:
                                residuals = residuals * np.array(
                                    [1 / norm if norm > 0 else 1 for norm in residuals.norm(product=self.products[k])]
                                )
                            new_vecs2[k].append(data["residuals"][k - 3])
            old_t = t

            if check_against_pickled:
                np.set_printoptions(precision=15)
                for k in range(6):
                    for j in range(len(new_vecs[k])):
                        vec1 = new_vecs[k]._list[j].to_numpy()
                        vec2 = new_vecs2[k]._list[j].to_numpy()
                        for z in range(new_vecs[k]._list[j].dim):
                            if not np.isclose(vec1[z], vec2[z], rtol=check_rtol, atol=check_atol):
                                print(
                                    f"{vec1[z]} vs. {vec2[z]}, err: { vec1[z] - vec2[z] }, relerr: { (vec1[z] - vec2[z]) / max(vec1[z], vec2[z]) }",
                                    flush=True,
                                )
                                raise AssertionError

            yield new_vecs

    def chunk_index(self):
        return self.current_chunk_index

    def done(self):
        return self.chunk_index() == (self.num_chunks - 1)


# exemplary call: mpiexec -n 2 python3 cellmodel_hapod_deim.py single_cell 1e-2 1e-3 30 30 True True True True True False 1e-3 1e-3 1e-3 1e-10 1e-10 1e-10
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
    parameter_sampling_type = "log_reciprocal" if argc < 19 else sys.argv[18]
    pod_method = "method_of_snapshots" if argc < 20 else sys.argv[19]
    assert pod_method in ("qr_svd", "method_of_snapshots")
    normalize_residuals = False
    incremental_gramian = False
    pol_order = 2
    chunk_size = 10
    visualize_step = 50
    pod_pfield = True
    pod_ofield = True
    pod_stokes = True
    least_squares_pfield = True
    least_squares_ofield = True
    least_squares_stokes = True
    if not pod_pfield:
        least_squares_pfield = False
    if not pod_ofield:
        least_squares_ofield = False
    if not pod_stokes:
        least_squares_stokes = False
    excluded_params = ("Pa", "Ca")
    # product_type = "L2"
    product_type = "H1"
    train_params_per_rank = 64
    test_params_per_rank = 1
    # omega=0.5
    omega = 0.95
    random.seed(123)  # create_parameters choose some parameters randomly in some cases

    ###### Choose filename #########
    logfile_dir = "logs"
    if mpi.rank_world == 0:
        if not os.path.exists(logfile_dir):
            os.mkdir(logfile_dir)
    logfile_prefix = "results_{}{}_{}_{}_{}procs_{}_grid{}x{}_tend{}_dt{}_{}_{}tppr_pfield{}_ofield{}_stokes{}_without".format(
        "normalized_" if normalize_residuals else "",
        "mos" if pod_method == "method_of_snapshots" else "qr_svd",
        parameter_sampling_type,
        testcase,
        mpi.size_world,
        product_type,
        grid_size_x,
        grid_size_y,
        t_end,
        dt,
        "snapsandstages" if include_newton_stages else "snaps",
        train_params_per_rank,
        (f"pod{pfield_atol:.0e}" if pod_pfield else "pod0")
        + (f"deim{pfield_deim_atol:.1e}" if deim_pfield else "deim0"),
        (f"pod{ofield_atol:.0e}" if pod_ofield else "pod0")
        + (f"deim{ofield_deim_atol:.1e}" if deim_ofield else "deim0"),
        (f"pod{stokes_atol:.0e}" if pod_stokes else "pod0")
        + (f"deim{stokes_deim_atol:.1e}" if deim_stokes else "deim0"),
    )
    for excluded_param in excluded_params:
        logfile_prefix += "_" + excluded_param
    logfile_prefix += f"_omega{omega}"
    logfile_name = os.path.join(logfile_dir, logfile_prefix + ".txt")

    ####### Collect some settings in lists for simpler handling #####
    hapod_tols = [pfield_atol, ofield_atol, stokes_atol, pfield_deim_atol, ofield_deim_atol, stokes_deim_atol]
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
    rf = 5  # Factor of largest to smallest training parameter
    mus, new_mus = create_parameters(
        train_params_per_rank,
        test_params_per_rank,
        rf,
        mpi,
        excluded_params,
        logfile_name,
        Be0=1.0,
        Ca0=1.0,
        Pa0=1.0,
        sampling_type=parameter_sampling_type,
    )

    ################### Start HAPOD #####################
    # only needed if check_against_pickled is True
    pickle_dir = "pickle_files"
    pickle_prefix = f"{testcase}_grid{grid_size_x}x{grid_size_y}_tend{t_end}_dt{dt}"
    pickle_prefix = os.path.join(pickle_dir, pickle_prefix)

    # create solver
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
    m = DuneCellModel(solver, products={"pfield": products[0], "ofield": products[1], "stokes": products[2]})

    # create chunk_generator
    chunk_generator = SolverChunkGenerator(
        cellmodel=m,
        pickle_prefix=pickle_prefix,
        t_end=t_end,
        dt=dt,
        mus=mus,
        chunk_size=chunk_size,
        include_newton_stages=include_newton_stages,
        indices=indices,
        products=products,
        normalize_residuals=normalize_residuals,
    )
    # perform HAPOD
    results = binary_tree_hapod(
        chunk_generator=chunk_generator,
        mpi=mpi,
        tolerances=hapod_tols,
        indices=indices,
        omega=0.95,
        logfile=logfile_name,
        products=products,
    )
    for k in indices:
        r = results[k]
        r.modes, r.win = mpi.shared_memory_bcast_modes(r.modes, returnlistvectorarray=True, proc_rank=k % mpi.size_proc)

    pfield_basis = results[0].modes if pod_pfield else None
    ofield_basis = results[1].modes if pod_ofield else None
    stokes_basis = results[2].modes if pod_stokes else None
    pfield_deim_basis = results[3].modes if deim_pfield else None
    ofield_deim_basis = results[4].modes if deim_ofield else None
    stokes_deim_basis = results[5].modes if deim_stokes else None

    reduced_prefix = "without{}_{}_pfield_{}_ofield_{}_stokes_{}".format(
        excluded_params,
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
    mpi.comm_world.Barrier()
    # time.sleep(10)
    us = []
    for mu in mus:
        u, _ = rom.solve(mu, return_stages=False)
        us.append(u)

    # U_rom = reductor.reconstruct(us)
    # Be, Ca, Pa = (float(mus[0]["Be"]), float(mus[0]["Ca"]), float(mus[0]["Pa"]))
    # m.visualize(
    #     U_rom,
    #     prefix=f"{filename[:-4]}_Be{Be}_Ca{Ca}_Pa{Pa}",
    #     subsampling=subsampling,
    #     every_nth=visualize_step,
    # )
    # del m
    # del solver

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
        logfile_name=logfile_name,
        products=products,
        pickled_data_available=False,
        num_chunks=chunk_generator.num_chunks,
    )

    ################## test new parameters #######################
    # solve full-order model for new param
    # start = timer()
    # U_new_mu = m.solve(mu=new_mus[0], return_stages=False)
    # for p in range(1, len(new_mus)):
    # U_new_mu.append(m.solve(mu=new_mus[p], return_stages=False))
    # mean_fom_time = (timer() - start) / len(new_mus)
    # Be, Ca, Pa = (float(new_mu['Be']), float(new_mu['Ca']), float(new_mu['Pa']))
    # m.visualize(U_new_mu, prefix=f"fullorder_Be{Be}_Ca{Ca}_Pa{Pa}", subsampling=subsampling, every_nth=visualize_step)

    # solve reduced model for new params
    # start = timer()
    # # cProfile.run(
    # #     "u_new_mu = rom.solve(new_mus[0], return_stages=False)", f"rom{mpi.rank_world}.cprof"
    # # )
    # # us_new_mu = []
    # for new_mu in new_mus:
    #     u, _ = rom.solve(new_mu, return_stages=False)
    #     us_new_mu.append(u)
    # mean_rom_time = (timer() - start) / len(new_mus)

    # Be, Ca, Pa = (float(new_mu['Be']), float(new_mu['Ca']), float(new_mu['Pa']))
    # m.visualize(
    #     U_rom_new_mu,
    #     prefix=f"{reduced_prefix}_Be{Be}_Ca{Ca}_Pa{Pa}",
    #     subsampling=subsampling,
    #     every_nth=visualize_step)

    # calculate_cellmodel_errors(
    #     modes=[pfield_basis, ofield_basis, stokes_basis],
    #     deim_modes=[pfield_deim_basis, ofield_deim_basis, stokes_deim_basis],
    #     testcase=testcase,
    #     t_end=t_end,
    #     dt=dt,
    #     grid_size_x=grid_size_x,
    #     grid_size_y=grid_size_y,
    #     pol_order=pol_order,
    #     mus=new_mus,
    #     reduced_us=us_new_mu,
    #     reductor=reductor,
    #     mpi_wrapper=mpi,
    #     logfile_name=logfile_name,
    #     prefix="new ",
    #     products=products,
    #     pickled_data_available=False,
    #     rom_time=mean_rom_time,
    # )

    sys.stdout.flush()
    mpi.comm_world.Barrier()
