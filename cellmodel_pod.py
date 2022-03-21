import resource
import sys

# import cProfile
import os
import random
import sys
import math
from timeit import default_timer as timer
from typing import Dict, Any

from pymor.algorithms.pod import pod

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
    calculate_cellmodel_errors,
    create_parameters,
    solver_statistics,
)
from hapod.mpi import MPIWrapper
from rich import pretty, traceback
from pymor.vectorarrays.numpy import NumpyVectorSpace
import numpy as np

traceback.install()
pretty.install()

def calculate_pod(result, product, mpi, tol, num_modes_equal):
    num_snapshots = len(result)

    # gather snapshots on rank 0
    result, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(
        result, num_snapshots, num_modes_equal=num_modes_equal
    )


    # perform a POD
    elapsed_pod = 0
    start = timer()
    svals = None
    if mpi.rank_world == 0:
        result, svals = pod(result, product=product, atol=0.0, rtol=0.0, l2_err=tol * math.sqrt(len(result)))
        elapsed_pod = timer() - start
    return result, svals, elapsed_pod, total_num_snapshots


def cellmodel_pod(m, mu, tols, logfile=None):
    # get MPI communicators
    mpi = MPIWrapper()

    # calculate Boltzmann problem trajectory
    start = timer()
    snapshots, data = m._compute_solution(mu, return_residuals=True)
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start

    modes, svals, elapsed_pod, total_num_snaps = [[None] * 6 for i in range(4)]
    # for the phase field, we compute separate bases for each variable
    modes[3], svals[3], elapsed_pod[3] = [[None] * 3 for i in range(3)]
    # Compute POD bases
    for k in range(3):
        modes[k], svals[k], elapsed_pod[k], total_num_snaps[k] = calculate_pod(
            snapshots._blocks[k], None, mpi, tols[k], True
        )
    # Compute phase field DEIM bases
    # vecs_field = [vecs.empty()] * 3
    size_phi = data["residuals"][0].dim // 3
    for i in range(3):
        modes[3][i] = NumpyVectorSpace.make_array(np.ascontiguousarray(data["residuals"][0].to_numpy()[:, i * size_phi : (i + 1) * size_phi]))
        modes[3][i], svals[3][i], elapsed_pod[3][i], total_num_snaps[3] = calculate_pod(
                modes[3][i], None, mpi, tols[3], False
            )
    # Compute orientation field and Stokes DEIM bases
    for k in range(4, 6):
        modes[k], svals[k], elapsed_pod[k], total_num_snaps[k] = calculate_pod(
            data["residuals"][k-3], None, mpi, tols[k], False
        )

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        with open(logfile, "a") as ff:
            ff.write(
                    "Pfield POD took {:.2f} s and resulted in {} modes from {} snapshots!\n".format(
                    elapsed_pod[0], len(modes[0]), total_num_snaps[0]
                )
            )
            ff.write(
                    "Ofield POD took {:.2f} s and resulted in {} modes from {} snapshots!\n".format(
                    elapsed_pod[1], len(modes[1]), total_num_snaps[1]
                )
            )
            ff.write(
                    "Stokes POD took {:.2f} s and resulted in {} modes from {} snapshots!\n".format(
                    elapsed_pod[2], len(modes[2]), total_num_snaps[2]
                )
            )
            ff.write(
                    "Phi (first phasefield variable) DEIM POD took {:.2f} s and resulted in {} modes from {} residuals!\n".format(
                    elapsed_pod[3][0], len(modes[3][0]), total_num_snaps[3]
                )
            )
            ff.write(
                    "Phinat (second phasefield variable) DEIM POD took {:.2f} s and resulted in {} modes from {} residuals!\n".format(
                    elapsed_pod[3][1], len(modes[3][1]), total_num_snaps[3]
                )
            )
            ff.write(
                    "Mu (third phasefield variable) DEIM POD took {:.2f} s and resulted in {} modes from {} residuals!\n".format(
                    elapsed_pod[3][2], len(modes[3][2]), total_num_snaps[3]
                )
            )
            ff.write(
                    "Ofield DEIM POD took {:.2f} s and resulted in {} modes from {} residuals!\n".format(
                    elapsed_pod[4], len(modes[4]), total_num_snaps[4]
                )
            )
            ff.write(
                    "Stokes DEIM POD took {:.2f} s and resulted in {} modes from {} residuals!\n".format(
                    elapsed_pod[5], len(modes[5]), total_num_snaps[5]
                )
            )
            ff.write(
                "The maximum amount of memory used on rank 0 was: "
                + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 ** 2)
                + " GB\n"
            )
            elapsed = timer() - start
            ff.write("Time elapsed: " + str(elapsed) + "\n")

    return [modes, svals, total_num_snaps, mu, mpi, solver]


if __name__ == "__main__":
    mpi = MPIWrapper()
    ##### read command line arguments, additional settings #####
    argc = len(sys.argv)
    testcase = "single_cell" if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 30 if argc < 5 else int(sys.argv[4])
    grid_size_y = 30 if argc < 6 else int(sys.argv[5])
    pfield_atol = 1e-3 if argc < 7 else float(sys.argv[6])
    ofield_atol = 1e-3 if argc < 8 else float(sys.argv[7])
    stokes_atol = 1e-3 if argc < 9 else float(sys.argv[8])
    pfield_deim_atol = 1e-10 if argc < 10 else float(sys.argv[9])
    ofield_deim_atol = 1e-10 if argc < 11 else float(sys.argv[10])
    stokes_deim_atol = 1e-10 if argc < 12 else float(sys.argv[11])
    parameter_sampling_type = "log_and_log_inverted" if argc < 13 else sys.argv[12]
    pod_method = "method_of_snapshots" if argc < 14 else sys.argv[13]
    assert pod_method in ("qr_svd", "method_of_snapshots")
    pol_order = 2
    excluded_params = ("Be", "Ca")
    # product_type = "L2"
    # product_type = "H1"
    product_type = "l2"
    train_params_per_rank = 1
    test_params_per_rank = 1
    random.seed(123)  # create_parameters choose some parameters randomly in some cases
    if train_params_per_rank > 1:
        raise NotImplementedError

    ###### Choose filename #########
    logfile_dir = "logs"
    if mpi.rank_world == 0:
        if not os.path.exists(logfile_dir):
            os.mkdir(logfile_dir)
    logfile_prefix = "results_pod_{}_{}_{}_{}procs_{}_grid{}x{}_tend{}_dt{}_{}_{}tppr_pfield{}_ofield{}_stokes{}_without".format(
        "mos" if pod_method == "method_of_snapshots" else "qr_svd",
        parameter_sampling_type,
        testcase,
        mpi.size_world,
        product_type,
        grid_size_x,
        grid_size_y,
        t_end,
        dt,
        train_params_per_rank,
        f"pod{pfield_atol:.0e}" ,
        f"deim{pfield_deim_atol:.1e}" ,
        f"pod{ofield_atol:.0e}" ,
        f"deim{ofield_deim_atol:.1e}" ,
        f"pod{stokes_atol:.0e}" ,
        f"deim{stokes_deim_atol:.1e}" ,
    )
    for excluded_param in excluded_params:
        logfile_prefix += "_" + excluded_param
    logfile_name = os.path.join(logfile_dir, logfile_prefix + ".txt")

    ####### Collect some settings in lists for simpler handling #####
    tols = [pfield_atol, ofield_atol, stokes_atol, pfield_deim_atol, ofield_deim_atol, stokes_deim_atol]

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
            CellModelOfieldH1ProductOperator(solver),
            CellModelStokesL2ProductOperator(solver),
        ] * 2
    else:
        products = [None] * 6
    m = DuneCellModel(solver, products={"pfield": products[0], "ofield": products[1], "stokes": products[2]})

    # perform POD
    modes, _, total_num_snaps, mu, mpi, solver = cellmodel_pod(m, mus[0], tols, logfile=logfile_name)
    products = [None] * 6

    wins = [None] * 6
    pfield_wins = [None] * 3
    for k in range(6):
        if k == 3:
            for i in range(3):
                modes[3][i], pfield_wins[i] = mpi.shared_memory_bcast_modes(
                    modes[3][i], returnlistvectorarray=True, proc_rank=0
                )
        else:
            modes[k], wins[k] = mpi.shared_memory_bcast_modes(
                modes[k], returnlistvectorarray=True, proc_rank=0
            )

    pfield_basis = modes[0]
    ofield_basis = modes[1]
    stokes_basis = modes[2]
    pfield_deim_basis = modes[3]
    ofield_deim_basis = modes[4]
    stokes_deim_basis = modes[5]

    reductor = CellModelReductor(
        m,
        pfield_basis,
        ofield_basis,
        stokes_basis,
        least_squares_pfield=True,
        least_squares_ofield=True,
        least_squares_stokes=True,
        pfield_deim_basis=pfield_deim_basis,
        ofield_deim_basis=ofield_deim_basis,
        stokes_deim_basis=stokes_deim_basis,
        check_orthonormality=True,
        check_tol=1e-10,
        products={"pfield": products[0], "ofield": products[1], "stokes": products[2]},
    )
    rom = reductor.reduce()

    ################## solve reduced model for trained parameters ####################
    mpi.comm_world.Barrier()
    # time.sleep(10)
    us = []
    for mu in mus:
        u, _ = rom.solve(mu, return_stages=False)
        us.append(u)

    mpi.comm_world.Barrier()
    calculate_cellmodel_errors(
        m,
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
    )

    ################## test new parameters #######################
    # solve full-order model for new param
    start = timer()
    for mu in new_mus:
        U_new_mu, _ = m.solve(mu=mu, return_stages=False)
    mean_fom_time = (timer() - start) / len(new_mus)
    del U_new_mu

    # solve reduced model for new params
    start = timer()
    us_new_mu = []
    for mu in new_mus:
        u, _ = rom.solve(mu, return_stages=False)
        # cProfile.run(
        #     "u = rom.solve(mu, return_stages=False)", f"rom{mpi.rank_world}.cprof"
        # )
        us_new_mu.append(u)
    mean_rom_time = (timer() - start) / len(new_mus)

    calculate_cellmodel_errors(
        m,
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
        logfile_name=logfile_name,
        prefix="new ",
        products=products,
        rom_time=mean_rom_time,
        fom_time=mean_fom_time,
    )

    sys.stdout.flush()
    mpi.comm_world.Barrier()
