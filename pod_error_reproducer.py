import pickle
import numpy as np
import scipy.io
import scipy.sparse

from pymor.algorithms.pod import pod
from pymor.operators.basic import OperatorBase
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray

class MatrixOperator(OperatorBase):

    def __init__(self, mat):
        self.mat = scipy.sparse.block_diag((mat, mat, mat), 'csr')
        self.solution_space = NumpyVectorSpace(self.mat.shape[0])

    def apply(self, U, mu=None):
        U_out = np.transpose(self.mat @ np.transpose(U.to_numpy()))
        return self.solution_space.make_array(U_out)

if __name__ == "__main__":

    with open("snaps_pfield.pickle", "rb") as f:
        snaps = pickle.load(f)
    snaps = NumpyVectorArray(snaps, NumpyVectorSpace(snaps.shape[1]))
    product = MatrixOperator(scipy.io.mmread("phasefield_mass_matrix.mtx").tocsr())
    tol = 1e-4
    modes, svals = pod(snaps, product=product, atol=0., rtol=0., l2_err=tol)
    residual = snaps - modes.lincomb(snaps.dot(product.apply(modes)))
    error_pfield = np.sqrt(np.sum(residual.pairwise_dot(product.apply(residual))))
    print("tol: {}, error: {}".format(tol, error_pfield))
