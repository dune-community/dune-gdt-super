import time
import numpy as np
from mpi4py import MPI

from pymor.vectorarrays.numpy import NumpyVectorSpace

from hapod.xt import DuneXtLaListVectorSpace

# The POD only uses MPI rank 0, all other processes are busy-waiting for it to finish.
# With idle_wait, the waiting threads do not cause such a high CPU usage.
# Adapted from https://gist.github.com/donkirkby/16a89d276e46abb0a106
def idle_wait(comm, root=0):
    # print(f"Rank {comm.rank} in idle_wait for root {root}", flush=True)
    if comm.rank == root:
        for other_rank in range(0, comm.size):
            if other_rank != root:
                # print(f"Send to rank {other_rank} from rank {comm.rank}", flush=True)
                comm.send(0, dest=other_rank, tag=other_rank)
    else:
        # Set this to 0 for maximum responsiveness, but that will peg CPU to 100%
        sleep_seconds = 0.1
        if sleep_seconds > 0:
            while not comm.Iprobe(source=root):
                # print(f"{comm.rank} waited another second for root {root}", flush=True)
                time.sleep(sleep_seconds)
        comm.recv(source=root, tag=comm.rank)

class MPIWrapper:
    """Stores MPI communicators for all ranks (world), for all ranks on a single compute node (proc)
       and for all ranks that have rank the same rank on their compute node (rank_group).
       Further provides some convenience functions for using MPI for VectorArrays"""

    def __init__(self):
        # Preparation: setup MPI
        # create world communicator
        self.comm_world = BoltzmannMPICommunicator(MPI.COMM_WORLD)
        self.rank_world = self.comm_world.rank
        self.size_world = self.comm_world.size

        # use processor numbers to create a communicator on each processor
        self.comm_proc = BoltzmannMPICommunicator(
            self.comm_world.Split_type(MPI.COMM_TYPE_SHARED, self.rank_world)
        )
        self.size_proc = self.comm_proc.size
        self.rank_proc = self.comm_proc.rank

        # create communicator containing rank k processes from each processor
        self.comm_rank_group = []
        self.size_rank_group = []
        self.rank_rank_group = []
        for k in range(self.size_proc):
            contained_in_rank_k_group = 1 if self.rank_proc == k else 0
            self.comm_rank_group.append(
                BoltzmannMPICommunicator(
                    self.comm_world.Split(contained_in_rank_k_group, self.rank_world)
                )
            )
            self.size_rank_group.append(self.comm_rank_group[k].size)
            self.rank_rank_group.append(self.comm_rank_group[k].rank)

    def shared_memory_bcast_modes(self, modes, returnlistvectorarray=False, proc_rank=0):
        """
        Broadcast modes on root rank to all ranks by using a shared memory buffer on each node

        Parameters
        ----------
        modes: ListVectorArray of (HA)POD modes
        returnlistvectorarray: If True, a DuneXtLaListVectorArray is returned instead of a NumpyVectorArray.

        Returns
        -----
        A tuple (modes, win) where modes is a DuneXtLaListVectorArray (if returnlistvectorarray=True) or a NumpyVectorArray
        (if returnlistvectorarray=False) containing the modes and win is the MPI window that holds the shared memory buffer.
        You have to free the memory yourself by calling win.Free() once you are done.
        """
        if modes is None:
            modes = np.empty(shape=(0, 0), dtype="d")
        self_is_root = self.rank_proc == proc_rank and self.rank_rank_group[proc_rank] == 0
        gathered_root_info = self.comm_world.gather(self_is_root, root=0)
        root = gathered_root_info.index(True) if self.rank_world == 0 else None
        root = self.comm_world.bcast(root, root=0)
        modes_length = self.comm_world.bcast(len(modes) if self_is_root else 0, root=root)
        assert modes_length > 0, "Cannot broadcast empty modes"
        vector_length = self.comm_world.bcast(modes[0].dim if self_is_root else 0, root=root)
        # create shared memory buffer to share final modes between processes on each node
        size = modes_length * vector_length
        itemsize = MPI.DOUBLE.Get_size()
        num_bytes = size * itemsize if self.rank_proc == proc_rank else 0
        win = MPI.Win.Allocate_shared(num_bytes, itemsize, comm=self.comm_proc)
        buf, itemsize = win.Shared_query(rank=proc_rank)
        assert itemsize == MPI.DOUBLE.Get_size()
        buf = np.array(buf, dtype="B", copy=False)
        modes_numpy = np.ndarray(buffer=buf, dtype="d", shape=(modes_length, vector_length))
        if self.rank_proc == proc_rank:
            if self_is_root:
                self.comm_rank_group[proc_rank].Bcast([modes.to_numpy(), MPI.DOUBLE], root=0)
                for i, v in enumerate(modes._list):
                    modes_numpy[i, :] = v.to_numpy()[:]
                    del v
            else:
                self.comm_rank_group[proc_rank].Bcast([modes_numpy, MPI.DOUBLE], root=0)
        self.comm_world.Barrier()  # without this barrier, non-zero ranks might be too fast
        if returnlistvectorarray:
            modes = DuneXtLaListVectorSpace.from_memory(modes_numpy)
        else:
            modes = NumpyVectorSpace.from_numpy(modes_numpy)
        return modes, win


class BoltzmannMPICommunicator(MPI.Intracomm):
    def __init__(self, comm):
        self.comm = comm

    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        comm = self.comm
        rank = comm.Get_rank()
        comm.send(
            [len(modes), len(svals) if svals is not None else 0, num_snaps_in_leafs, modes[0].dim],
            dest=dest,
            tag=rank + 1000,
        )
        comm.Send(modes.to_numpy(), dest=dest, tag=rank + 2000)
        if svals is not None:
            comm.Send(svals, dest=dest, tag=rank + 3000)

    def recv_modes(self, source):
        comm = self.comm
        len_modes, len_svals, total_num_snapshots, vector_length = comm.recv(
            source=source, tag=source + 1000
        )
        received_array = np.empty(shape=(len_modes, vector_length))
        comm.Recv(received_array, source=source, tag=source + 2000)
        modes = DuneXtLaListVectorSpace.from_numpy(received_array)
        svals = np.empty(shape=(len_modes,))
        if len_svals > 0:
            comm.Recv(svals, source=source, tag=source + 3000)

        return modes, svals, total_num_snapshots

    def gather_on_root_rank(
        self,
        vectorarray,
        num_snapshots_on_rank,
        svals=None,
        num_modes_equal=False,
        merge=True,
        root=0,
    ):
        comm = self.comm
        rank = comm.Get_rank()
        if svals is not None:
            assert len(svals) == len(vectorarray)
        num_snapshots_in_associated_leafs = comm.reduce(
            num_snapshots_on_rank, op=MPI.SUM, root=root
        )
        total_num_modes = comm.reduce(len(vectorarray), op=MPI.SUM, root=root)
        vector_length = vectorarray[0].dim if len(vectorarray) > 0 else 0
        # create empty numpy array on rank 0 as a buffer to receive the pod modes from each core
        vectors_gathered = (
            np.empty(shape=(total_num_modes, vector_length)) if rank == root else None
        )
        svals_gathered = (
            np.empty(shape=(total_num_modes,)) if (rank == root and svals is not None) else None
        )
        # gather the modes (as numpy array, thus the call to data) in vectors_gathered.
        offsets = []
        offsets_svals = []
        # if we have the same number of modes on each rank, we can use Gather, else we have to use Gatherv
        if num_modes_equal:
            comm.Gather(vectorarray.to_numpy(), vectors_gathered, root=root)
            if svals is not None:
                comm.Gather(svals, svals_gathered, root=root)
        else:
            # Gatherv needed because every process can send a different number of modes
            counts = comm.gather(len(vectorarray) * vector_length, root=root)
            if svals is not None:
                counts_svals = comm.gather(len(svals), root=root)
            if rank == root:
                offsets = [0]
                for j, count in enumerate(counts):
                    offsets.append(offsets[j] + count)
                comm.Gatherv(
                    vectorarray.to_numpy(),
                    [vectors_gathered, counts, offsets[0:-1], MPI.DOUBLE],
                    root=root,
                )
                if svals is not None:
                    offsets_svals = [0]
                    for j, count in enumerate(counts_svals):
                        offsets_svals.append(offsets_svals[j] + count)
                    comm.Gatherv(
                        svals,
                        [svals_gathered, counts_svals, offsets_svals[0:-1], MPI.DOUBLE],
                        root=root,
                    )
            else:
                comm.Gatherv(vectorarray.to_numpy(), None, root=root)
                if svals is not None:
                    comm.Gatherv(svals, None, root=root)
        del vectorarray
        if rank == root:
            if merge:
                vectors_gathered = DuneXtLaListVectorSpace.from_numpy(
                    vectors_gathered, ensure_copy=True
                )
            else:
                vectors = []
                for i in range(len(offsets) - 1):
                    vectors.append(
                        DuneXtLaListVectorSpace.from_numpy(
                            vectors_gathered[offsets_svals[i] : offsets_svals[i + 1]],
                            ensure_copy=True,
                        )
                    )
                vectors_gathered = vectors
                if svals is not None:
                    svals = []
                    for i in range(len(offsets_svals) - 1):
                        svals.append(svals_gathered[offsets_svals[i] : offsets_svals[i + 1]].copy())
                svals_gathered = svals
        return vectors_gathered, svals_gathered, num_snapshots_in_associated_leafs, offsets_svals

    def gather_on_rank_0(
        self, vectorarray, num_snapshots_on_rank, svals=None, num_modes_equal=False, merge=True
    ):
        return self.gather_on_root_rank(
            vectorarray=vectorarray, num_snapshots_on_rank=num_snapshots_on_rank, svals=svals, num_modes_equal=num_modes_equal, merge=merge, root=0
        )

    def __getattr__(self, item):
        """ Redirects all request for properties and methods that are not defined in this class to self.comm """
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self.comm.__getattr__(item)  # redirection
