import resource
import sys
from timeit import default_timer as timer

import numpy as np
from pymor.algorithms.pod import pod

from mpiwrapper import MPIWrapper
from boltzmann.wrapper import CellModelSolver, CellModelPfieldProductOperator, CellModelOfieldProductOperator, CellModelStokesProductOperator, calculate_cellmodel_error, create_and_scatter_cellmodel_parameters, DuneXtLaListVectorSpace
from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


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


def cellmodel_pod(testcase, t_end, dt, grid_size_x, grid_size_y, tol, logfile=None):
    # get MPI communicators
    mpi = MPIWrapper()

    # get cellmodel solver to create snapshots
    mu = create_and_scatter_cellmodel_parameters(mpi.comm_world)
    solver = CellModelSolver(testcase, t_end, grid_size_x, grid_size_y, mu)
    nc = solver.num_cells

    # calculate Boltzmann problem trajectory
    start = timer()
    snapshots = solver.solve(dt, True, dt, "result_" + str(mu))
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


# class MatrixOperator(OperatorBase):
#
#     def __init__(self, mat):
#         self.mat = scipy.sparse.block_diag((mat, mat, mat), 'csr')
#         self.mass_mat = mat
#         self.n = n = mat.shape[0]
#         assert (self.mat[0:n, 0:n] != mat).nnz == 0
#         assert (self.mat[n:2 * n, n:2 * n] != mat).nnz == 0
#         assert (self.mat[2 * n:3 * n, 2 * n:3 * n] != mat).nnz == 0
#         self.numpy_solution_space = NumpyVectorSpace(self.mat.shape[0])
#         self.solution_space = DuneXtLaListVectorSpace(self.mat.shape[0])
#
#     def apply(self, U, mu=None, res=None):
#         U_out = np.transpose(self.mat @ np.transpose(U.to_numpy()))
#         return self.solution_space.from_numpy(U_out)
#
#     def calculate_error(self, modes, name, product=None):
#         if product is None:
#             product = self
#         snaps = load_snapshots(name)
#         residual = snaps - modes.lincomb(
#             snaps.dot(self.numpy_solution_space.from_numpy(product.apply(modes).to_numpy())))
#         return np.sqrt(
#             np.sum(residual.pairwise_dot(self.numpy_solution_space.from_numpy(product.apply(residual).to_numpy()))))
#
#        # residual = snaps - modes.lincomb(snaps.dot(product.apply(modes, numpy=True)))
#        # return np.sqrt(np.sum(residual.pairwise_dot(product.apply(residual, numpy=True))))

if __name__ == "__main__":
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else float(sys.argv[2])
    dt = 1e-3 if argc < 4 else float(sys.argv[3])
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    tol = 1e-4 if argc < 7 else float(sys.argv[6])
    filename = "cellmodel_POD_grid_%dx%d_tol_%f" % (grid_size_x, grid_size_y, tol)
    logfile = open(filename, "a")
    modes, _, total_num_snaps, mu, mpi = cellmodel_pod(
        testcase, t_end, dt, grid_size_x, grid_size_y, tol, logfile=logfile)

    # Broadcast modes to all ranks, if possible via shared memory
    retlistvecs = True
    wins = total_num_snaps.copy()
    for i in range(len(modes)):
        modes[i], wins[i] = mpi.shared_memory_bcast_modes(modes[i], returnlistvectorarray=retlistvecs)

    # calculate errors
    errors = calculate_cellmodel_errors(modes, testcase, t_end, dt, grid_size_x, grid_size_y, mu, mpi, logfile=logfile)

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
