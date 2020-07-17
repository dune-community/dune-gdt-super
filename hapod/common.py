class MPICommunicator(object):

    rank = None
    size = None

    @abstractmethod
    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        pass

    @abstractmethod
    def recv_modes(self, source):
        pass
