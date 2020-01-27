from pymor.vectorarrays.numpy import NumpyVectorSpace
from mpi4py import MPI
import numpy as np

from pymor.core.base import abstractmethod

from hapod.xt import DuneXtLaListVectorSpace

class MPIWrapper:
    '''Stores MPI communicators for all ranks (world), for all ranks on a single compute node (proc)
       and for all ranks that have rank 0 on their compute node (rank_0_group).
       Further provides some convenience functions for using MPI for VectorArrays'''

    def __init__(self):
        # Preparation: setup MPI
        # create world communicator
        self.comm_world = BoltzmannMPICommunicator(MPI.COMM_WORLD)
        self.rank_world = self.comm_world.rank
        self.size_world = self.comm_world.size

        # use processor numbers to create a communicator on each processor
        self.comm_proc = BoltzmannMPICommunicator(self.comm_world.Split_type(MPI.COMM_TYPE_SHARED))
        self.size_proc = self.comm_proc.size
        self.rank_proc = self.comm_proc.rank

        # create communicator containing rank 0 processes from each processor
        self.contained_in_rank_0_group = 1 if self.rank_proc == 0 else 0
        self.comm_rank_0_group = BoltzmannMPICommunicator(
            self.comm_world.Split(self.contained_in_rank_0_group, self.rank_world))
        self.size_rank_0_group = self.comm_rank_0_group.size
        self.rank_rank_0_group = self.comm_rank_0_group.rank

    def shared_memory_bcast_modes(self, modes, returnlistvectorarray=False):
        ''' broadcast modes on rank 0 to all ranks by using a shared memory buffer on each node
            :param modes: ListVectorArray of (HA)POD modes
            :param returnlistvectorarray: If True, a DuneXtLaListVectorArray is returned instead of a
            NumpyVectorArray. On each node, the VectorArrays for all MPI ranks (one for each processor
            core) share the same underlying memory buffer.
            :returns: A tuple (modes, win) where modes is a DuneXtLaListVectorArray (if
            returnlistvectorarray=True) or a NumpyVectorArray (if returnlistvectorarray=False)
            containing the modes and win is the MPI window that holds the shared memory buffer. You
            have to free the memory yourself by calling win.Free() once you are done.'''
        if modes is None:
            modes = np.empty(shape=(0, 0), dtype='d')
        modes_length = self.comm_world.bcast(len(modes) if self.rank_world == 0 else 0, root=0)
        vector_length = self.comm_world.bcast(modes[0].dim if self.rank_world == 0 else 0, root=0)
        # create shared memory buffer to share final modes between processes on each node
        size = modes_length * vector_length
        itemsize = MPI.DOUBLE.Get_size()
        num_bytes = size * itemsize if self.rank_proc == 0 else 0
        win = MPI.Win.Allocate_shared(num_bytes, itemsize, comm=self.comm_proc)
        buf, itemsize = win.Shared_query(rank=0)
        assert itemsize == MPI.DOUBLE.Get_size()
        buf = np.array(buf, dtype='B', copy=False)
        modes_numpy = np.ndarray(buffer=buf, dtype='d', shape=(modes_length, vector_length))
        if self.rank_proc == 0:
            if self.rank_world == 0:
                self.comm_rank_0_group.comm.Bcast([modes.data, MPI.DOUBLE], root=0)
                for i, v in enumerate(modes._list):
                    modes_numpy[i, :] = v.data[:]
                    del v
            else:
                self.comm_rank_0_group.Bcast([modes_numpy, MPI.DOUBLE], root=0)
        self.comm_world.Barrier()     # without this barrier, non-zero ranks might be too fast
        if returnlistvectorarray:
            modes = DuneXtLaListVectorSpace.from_memory(modes_numpy)
            return modes, win
        else:
            modes = NumpyVectorSpace.from_numpy(modes_numpy)
            return modes, win

class MPICommunicator(object):

    rank = None
    size = None

    @abstractmethod
    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        pass

    @abstractmethod
    def recv_modes(self, source):
        pass

class BoltzmannMPICommunicator(MPICommunicator, MPI.Intracomm):

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        comm = self.comm
        rank = comm.Get_rank()
        comm.send([len(modes), len(svals) if svals is not None else 0, num_snaps_in_leafs, modes[0].dim],
                  dest=dest,
                  tag=rank + 1000)
        comm.Send(modes.data, dest=dest, tag=rank + 2000)
        if svals is not None:
            comm.Send(svals, dest=dest, tag=rank + 3000)

    def recv_modes(self, source):
        comm = self.comm
        len_modes, len_svals, total_num_snapshots, vector_length = comm.recv(source=source, tag=source + 1000)
        received_array = np.empty(shape=(len_modes, vector_length))
        comm.Recv(received_array, source=source, tag=source + 2000)
        modes = DuneXtLaListVectorSpace.from_numpy(received_array)
        svals = np.empty(shape=(len_modes,))
        if len_svals > 0:
            comm.Recv(svals, source=source, tag=source + 3000)

        return modes, svals, total_num_snapshots

    def gather_on_rank_0(self, vectorarray, num_snapshots_on_rank, svals=None, num_modes_equal=False):
        comm = self.comm
        rank = comm.Get_rank()
        if svals is not None:
            assert (len(svals) == len(vectorarray))
        num_snapshots_in_associated_leafs = comm.reduce(num_snapshots_on_rank, op=MPI.SUM, root=0)
        total_num_modes = comm.reduce(len(vectorarray), op=MPI.SUM, root=0)
        vector_length = vectorarray[0].dim if len(vectorarray) > 0 else 0
        # create empty numpy array on rank 0 as a buffer to receive the pod modes from each core
        vectors_gathered = np.empty(shape=(total_num_modes, vector_length)) if rank == 0 else None
        svals_gathered = np.empty(shape=(total_num_modes,)) if (rank == 0 and svals is not None) else None
        # gather the modes (as numpy array, thus the call to data) in vectors_gathered.
        offsets = []
        offsets_svals = []
        # if we have the same number of modes on each rank, we can use Gather, else we have to use Gatherv
        if num_modes_equal:
            comm.Gather(vectorarray.data, vectors_gathered, root=0)
            if svals is not None:
                comm.Gather(svals, svals_gathered, root=0)
        else:
            # Gatherv needed because every process can send a different number of modes
            counts = comm.gather(len(vectorarray) * vector_length, root=0)
            if svals is not None:
                counts_svals = comm.gather(len(vectorarray), root=0)
            if rank == 0:
                offsets = [0]
                for j, count in enumerate(counts):
                    offsets.append(offsets[j] + count)
                comm.Gatherv(vectorarray.data, [vectors_gathered, counts, offsets[0:-1], MPI.DOUBLE], root=0)
                if svals is not None:
                    offsets_svals = [0]
                    for j, count in enumerate(counts_svals):
                        offsets_svals.append(offsets_svals[j] + count)
                    comm.Gatherv(svals, [svals_gathered, counts_svals, offsets_svals[0:-1], MPI.DOUBLE], root=0)
            else:
                comm.Gatherv(vectorarray.data, None, root=0)
                if svals is not None:
                    comm.Gatherv(svals, None, root=0)
        del vectorarray
        if rank == 0:
            vectors_gathered = DuneXtLaListVectorSpace.from_numpy(vectors_gathered)
        return vectors_gathered, svals_gathered, num_snapshots_in_associated_leafs, offsets_svals

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self.comm.__getattr__(item)     # redirection
