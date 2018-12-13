from abc import abstractmethod
import numpy as np
from pymor.algorithms.pod import pod
from pymor.basic import gram_schmidt
from pymor.vectorarrays.interfaces import VectorArrayInterface
from scipy.linalg import eigh


class HapodParameters:
    '''Stores the HAPOD parameters :math:`\omega`, :math:`\epsilon^\ast` and :math:`L_\mathcal{T}` for easier passing
       and provides the local error tolerance :math:`\varepsilon_\mathcal{T}(\alpha)` '''
    def __init__(self, rooted_tree_depth, epsilon_ast=1e-4, omega=0.95):
        self.epsilon_ast = epsilon_ast
        self.omega = omega
        self.rooted_tree_depth = rooted_tree_depth

    def get_epsilon_alpha(self, num_snaps_in_leafs, root_of_tree=False):
        if not root_of_tree:
            epsilon_alpha = self.epsilon_ast * np.sqrt(1. - self.omega**2) * \
                            np.sqrt(num_snaps_in_leafs) / np.sqrt(self.rooted_tree_depth - 1)
        else:
            epsilon_alpha = self.epsilon_ast * self.omega * np.sqrt(num_snaps_in_leafs)
        return epsilon_alpha


def local_pod(inputs, num_snaps_in_leafs, parameters, root_of_tree=False, orthonormalize=True,
              incremental_gramian=True):
    '''Calculates a POD in the HAPOD tree. The input is a list where each element is either a vectorarray or
       a pair of (orthogonal) vectorarray and singular values from an earlier POD. If incremental_gramian is True, the
       algorithm avoids the recalculation of the diagonal blocks where possible by using the singular values.
       :param inputs: list of input vectors (and svals)
       :type inputs: list where each element is either a vectorarray or [vectorarray, numpy.ndarray]
       :param num_snaps_in_leafs: The number of snapshots below the current node (:math:`\widetilde{\mathcal{S}}_\alpha`)
       :param parameters: An object of type HapodParameters
       :param root_of_tree: Whether this is the root of the HAPOD tree
       :param orthonormalize: Whether to reorthonormalize the resulting modes
       :param incremental_gramian: Whether to build the gramian incrementally using information from the
       singular values'''
    # calculate offsets and check whether svals are provided in input
    offsets = [0]
    svals_provided = []
    vector_length = 0
    epsilon_alpha = parameters.get_epsilon_alpha(num_snaps_in_leafs, root_of_tree=root_of_tree)
    for i, modes in enumerate(inputs):
        if type(modes) is list:
            assert(issubclass(type(modes[0]), VectorArrayInterface))
            assert(issubclass(type(modes[1]), np.ndarray) and modes[1].ndim == 1)
            modes[0].scal(modes[1])
            svals_provided.append(True)
        elif issubclass(type(modes), VectorArrayInterface):
            inputs[i] = [modes]
            svals_provided.append(False)
        else:
            raise ValueError("")
        offsets.append(offsets[-1]+len(inputs[i][0]))
        vector_length = max(vector_length, inputs[i][0].dim)

    if incremental_gramian:
        # calculate gramian avoiding recalculations
        gramian = np.empty((offsets[-1],) * 2)
        all_modes = inputs[0][0].space.empty()
        for i in range(len(inputs)):
            modes_i, svals_i = [inputs[i][0], inputs[i][1] if svals_provided[i] else None]
            gramian[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = np.diag(svals_i)**2 if svals_provided[i] \
                                                                                            else modes_i.gramian()
            for j in range(i+1, len(inputs)):
                modes_j = inputs[j][0]
                cross_gramian = modes_i.dot(modes_j)
                gramian[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = cross_gramian
                gramian[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = cross_gramian.T
            all_modes.append(modes_i)
        modes_i._list = None

        EVALS, EVECS = eigh(gramian, overwrite_a=True, turbo=True, eigvals=None)
        del gramian

        EVALS = EVALS[::-1]
        EVECS = EVECS.T[::-1, :]  # is this a view? yes it is!

        errs = np.concatenate((np.cumsum(EVALS[::-1])[::-1], [0.]))

        below_err = np.where(errs <= epsilon_alpha**2)[0]
        first_below_err = below_err[0]

        svals = np.sqrt(EVALS[:first_below_err])
        EVECS = EVECS[:first_below_err]

        final_modes = all_modes.lincomb(EVECS / svals[:, np.newaxis])
        all_modes._list = None
        del modes
        del EVECS

        if orthonormalize:
            final_modes = gram_schmidt(final_modes, copy=False)

        return final_modes, svals
    else:
        modes = inputs[0][0].empty()
        for i in range(len(inputs)):
            modes.append(inputs[i][0])
        return pod(modes, atol=0., rtol=0., l2_err=epsilon_alpha, orthonormalize=orthonormalize, check=False)


class MPICommunicator(object):

    rank = None
    size = None

    @abstractmethod
    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        pass

    @abstractmethod
    def recv_modes(self, source):
        pass


def incremental_hapod_over_ranks(comm, modes, num_snaps_in_leafs, parameters, svals=None, last_hapod=False,
                                 incremental_gramian=True):
    ''' A incremental HAPOD with modes and possibly svals stored on ranks of the MPI communicator comm.
        May be used as part of a larger HAPOD tree, in that case you need to specify whether this
        part of the tree contains the root node (last_hapod=True)'''
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0

    if comm.size > 1:
        for current_rank in range(1, comm.size):
            # send modes and svals to rank 0
            if comm.rank == current_rank:
                comm.send_modes(0, modes, svals, num_snaps_in_leafs)
                modes = None
            # receive modes and svals
            elif comm.rank == 0:
                modes_on_source, svals_on_source, total_num_snapshots_on_source = comm.recv_modes(current_rank)
                max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len(modes_on_source))
                total_num_snapshots += total_num_snapshots_on_source
                modes, svals = local_pod([[modes, svals],
                                          [modes_on_source, svals_on_source] if len(svals_on_source) > 0
                                          else modes_on_source],
                                         total_num_snapshots,
                                         parameters,
                                         incremental_gramian=incremental_gramian,
                                         root_of_tree=(current_rank == comm.size - 1 and last_hapod))
                max_local_modes = max(max_local_modes, len(modes))
                del modes_on_source
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes


def binary_tree_depth(comm):
    """Calculates depth of binary tree of MPI ranks"""
    binary_tree_depth = 1
    ranks = range(0, comm.size)
    while len(ranks) > 1:
        binary_tree_depth += 1
        remaining_ranks = list(ranks)
        for odd_index in range(1, len(ranks), 2):
            remaining_ranks.remove(ranks[odd_index])
        ranks = remaining_ranks
    return binary_tree_depth


def binary_tree_hapod_over_ranks(comm, modes, num_snaps_in_leafs, parameters, svals=None, last_hapod=False,
                                 incremental_gramian=True):
    ''' A HAPOD with modes and possibly svals stored on ranks of the MPI communicator comm. A binary tree
        of MPI ranks is used as HAPOD tree.
        May be used as part of a larger HAPOD tree, in that case you need to specify whether this
        part of the tree contains the root node (last_hapod=True) '''
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0
    if comm.size > 1:
        ranks = range(0, comm.size)
        while len(ranks) > 1:
            remaining_ranks = list(ranks)
            # nodes with odd index send data to the node with index-1 where the pod is performed
            # this ensures that the modes end up on rank 0 in the end
            for odd_index in range(1, len(ranks), 2):
                sending_rank = ranks[odd_index]
                receiving_rank = ranks[odd_index-1]
                remaining_ranks.remove(sending_rank)
                if comm.rank == sending_rank:
                    comm.send_modes(receiving_rank, modes, svals, total_num_snapshots)
                    modes = None
                elif comm.rank == receiving_rank:
                    modes_on_source, svals_on_source, total_num_snapshots_on_source = comm.recv_modes(sending_rank)
                    max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len(modes_on_source))
                    total_num_snapshots += total_num_snapshots_on_source
                    modes, svals = local_pod([[modes, svals],
                                              [modes_on_source, svals_on_source] if len(svals_on_source) > 0
                                              else modes_on_source],
                                             total_num_snapshots,
                                             parameters,
                                             incremental_gramian=incremental_gramian,
                                             root_of_tree=((len(ranks) == 2) and last_hapod))
                    max_local_modes = max(max_local_modes, len(modes))
            ranks = list(remaining_ranks)
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes
