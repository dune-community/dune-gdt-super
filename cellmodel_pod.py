import resource
import sys
from timeit import default_timer as timer

from pymor.algorithms.pod import pod

from hapod.mpi import MPIWrapper
from hapod.cellmodel.wrapper import CellModelSolver, CellModelPfieldProductOperator, CellModelOfieldProductOperator, CellModelStokesProductOperator, calculate_cellmodel_errors, create_and_scatter_cellmodel_parameters

def calculate_pod(result, product, mpi, tol, name):
    num_snapshots = len(result)

    # gather snapshots on rank 0
    result, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(result, num_snapshots, num_modes_equal=True)

    # if mpi.rank_world == 0:
    #     with open("input_vectors_{}.pickle".format(name), "wb") as f:
    #         pickle.dump(result.to_numpy(), f)
    svals = None

    # perform a POD
    elapsed_pod = 0
    start = timer()
    if mpi.rank_world == 0:
        result, svals = pod(result, product=product, atol=0., rtol=0., l2_err=tol)
        elapsed_pod = timer() - start
    return result, svals, elapsed_pod, total_num_snapshots


def cellmodel_pod(testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, tol, logfile=None):
    # get MPI communicators
    mpi = MPIWrapper()

    # get cellmodel solver to create snapshots
    mu = create_and_scatter_cellmodel_parameters(mpi.comm_world)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, pol_order, mu)
    nc = solver.num_cells

    # calculate Boltzmann problem trajectory
    start = timer()
    snapshots = solver.solve(dt, True, dt, f"cellmodel_Be_{mu['Be']}_Ca_{mu['Ca']}_Pa_{mu['Pa']}")
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start

    # perform pod
    pfield_product = CellModelPfieldProductOperator(solver)
    ofield_product = CellModelOfieldProductOperator(solver)
    stokes_product = CellModelStokesProductOperator(solver)

    modes, svals, elapsed_pod, total_num_snaps = [[None] * (2 * nc + 1) for i in range(4)]
    for k in range(nc):
        modes[k], svals[k], elapsed_pod[k], total_num_snaps[k] = calculate_pod(snapshots[k], pfield_product, mpi, tol,
                                                                               "pfield_" + str(k))
        modes[nc + k], svals[nc + k], elapsed_pod[nc + k], total_num_snaps[nc + k] = calculate_pod(
            snapshots[nc + k], ofield_product, mpi, tol, "ofield_" + str(k))
    modes[2 * nc], svals[2 * nc], elapsed_pod[2 * nc], total_num_snaps[2 * nc] = calculate_pod(
        snapshots[2 * nc], stokes_product, mpi, tol, "stokes")

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        for k in range(nc):
            logfile.write("After the {}-th pfield POD, there are {} modes of {} snapshots left!\n".format(
                k, len(modes[k]), total_num_snaps[k]))
            logfile.write("After the {}-th ofield POD, there are {} modes of {} snapshots left!\n".format(
                k, len(modes[nc + k]), total_num_snaps[nc + k]))
        logfile.write("After the stokes POD, there are {} modes of {} snapshots left!\n".format(
            len(modes[2 * nc]), total_num_snaps[2 * nc]))
        logfile.write("The maximum amount of memory used on rank 0 was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB\n")
        elapsed = timer() - start
        logfile.write("Time elapsed: " + str(elapsed) + "\n")

    return [modes, svals, total_num_snaps, mu, mpi]


if __name__ == "__main__":
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    tol = 1e-4 if argc < 7 else float(sys.argv[6])
    filename = "cellmodel_POD_grid_%dx%d_tol_%f.log" % (grid_size_x, grid_size_y, tol)
    logfile = open(filename, "a")
    pol_order = 2
    modes, _, total_num_snaps, mu, mpi = cellmodel_pod(
        testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, tol, logfile=logfile)

    # Broadcast modes to all ranks, if possible via shared memory
    retlistvecs = True
    wins = total_num_snaps.copy()
    for i in range(len(modes)):
        modes[i], wins[i] = mpi.shared_memory_bcast_modes(modes[i], returnlistvectorarray=retlistvecs)

    # calculate errors
    errors = calculate_cellmodel_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, pol_order, mu, mpi, logfile=logfile)

    # free MPI windows
    for win in wins:
        win.Free()
    mpi.comm_world.Barrier()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
