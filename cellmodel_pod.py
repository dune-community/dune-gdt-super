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

    # calculate Boltzmann problem trajectory
    start = timer()
    snapshots_pfield, snapshots_ofield, snapshots_stokes = solver.solve(dt, True, dt, "result_" + str(mu))
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start

    # perform pod
    pfield_product = CellModelPfieldProductOperator(solver)
    ofield_product = CellModelOfieldProductOperator(solver)
    stokes_product = CellModelStokesProductOperator(solver)

    [result_pfield, svals_pfield, elapsed_pod_pfield, total_num_snapshots_pfield] = calculate_pod(
        snapshots_pfield, pfield_product, mpi, tol, "pfield")
    [result_ofield, svals_ofield, elapsed_pod_ofield, total_num_snapshots_ofield] = calculate_pod(
        snapshots_ofield, ofield_product, mpi, tol, "ofield")
    [result_stokes, svals_stokes, elapsed_pod_stokes, total_num_snapshots_stokes] = calculate_pod(
        snapshots_stokes, stokes_product, mpi, tol, "stokes")

    # write statistics to file
    if logfile is not None and mpi.rank_world == 0:
        logfile.write("After the pfield POD, there are " + str(len(result_pfield)) + " modes of " +
                      str(total_num_snapshots_pfield) + " snapshots left!\n")
        logfile.write("After the ofield POD, there are " + str(len(result_ofield)) + " modes of " +
                      str(total_num_snapshots_ofield) + " snapshots left!\n")
        logfile.write("After the stokes POD, there are " + str(len(result_stokes)) + " modes of " +
                      str(total_num_snapshots_stokes) + " snapshots left!\n")
        logfile.write("The maximum amount of memory used on rank 0 was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB\n")
        elapsed = timer() - start
        logfile.write("Time elapsed: " + str(elapsed) + "\n")

    return [
        result_pfield, svals_pfield, total_num_snapshots_pfield, result_ofield, svals_ofield,
        total_num_snapshots_pfield, result_stokes, svals_stokes, total_num_snapshots_stokes, mu, mpi
    ]


class MatrixOperator(OperatorBase):

    def __init__(self, mat):
        self.mat = scipy.sparse.block_diag((mat, mat, mat), 'csr')
        self.mass_mat = mat
        self.n = n = mat.shape[0]
        assert (self.mat[0:n, 0:n] != mat).nnz == 0
        assert (self.mat[n:2 * n, n:2 * n] != mat).nnz == 0
        assert (self.mat[2 * n:3 * n, 2 * n:3 * n] != mat).nnz == 0
        self.numpy_solution_space = NumpyVectorSpace(self.mat.shape[0])
        self.solution_space = DuneXtLaListVectorSpace(self.mat.shape[0])

    def apply(self, U, mu=None, res=None):
        U_out = np.transpose(self.mat @ np.transpose(U.to_numpy()))
        return self.solution_space.from_numpy(U_out)

    def calculate_error(self, modes, name, product=None):
        if product is None:
            product = self
        snaps = load_snapshots(name)
        residual = snaps - modes.lincomb(
            snaps.dot(self.numpy_solution_space.from_numpy(product.apply(modes).to_numpy())))
        return np.sqrt(
            np.sum(residual.pairwise_dot(self.numpy_solution_space.from_numpy(product.apply(residual).to_numpy()))))

        # residual = snaps - modes.lincomb(snaps.dot(product.apply(modes, numpy=True)))
        # return np.sqrt(np.sum(residual.pairwise_dot(product.apply(residual, numpy=True))))


if __name__ == "__main__":
    argc = len(sys.argv)
    testcase = 'single_cell' if argc < 2 else sys.argv[1]
    t_end = 1e-2 if argc < 3 else sys.argv[2]
    dt = 1e-3 if argc < 4 else sys.argv[3]
    grid_size_x = 20 if argc < 5 else int(sys.argv[4])
    grid_size_y = 5 if argc < 6 else int(sys.argv[5])
    tol = 1e-4 if argc < 7 else float(sys.argv[6])
    filename = "cellmodel_POD_grid_%dx%d_tol_%f" % (grid_size_x, grid_size_y, tol)
    logfile = open(filename, "a")
    [
        final_modes_pfield, _, total_num_snapshots_pfield, final_modes_ofield, _, total_num_snapshots_ofield,
        final_modes_stokes, _, total_num_snapshots_stokes, mu, mpi
    ] = cellmodel_pod(
        testcase, t_end, dt, grid_size_x, grid_size_y, tol, logfile=logfile)

    retlistvecs = True
    final_modes_pfield, win_pfield = mpi.shared_memory_bcast_modes(
        final_modes_pfield, returnlistvectorarray=retlistvecs)
    final_modes_ofield, win_ofield = mpi.shared_memory_bcast_modes(
        final_modes_ofield, returnlistvectorarray=retlistvecs)
    final_modes_stokes, win_stokes = mpi.shared_memory_bcast_modes(
        final_modes_stokes, returnlistvectorarray=retlistvecs)

    err_pfield, err_ofield, err_stokes = calculate_cellmodel_error(
        final_modes_pfield,
        final_modes_ofield,
        final_modes_stokes,
        testcase,
        t_end,
        dt,
        grid_size_x,
        grid_size_y,
        mu,
        mpi,
        logfile=logfile)
    win_pfield.Free()
    win_ofield.Free()
    win_stokes.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
