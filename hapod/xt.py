import numpy as np

from pymor.core.base import abstractmethod
from pymor.vectorarrays.list import Vector, ListVectorSpace, ListVectorArray

from gdt.vectors import CommonDenseVector


class DuneXtLaVector(Vector):
    def __init__(self, impl):
        self.impl = impl
        self.dim = impl.dim

    def to_numpy(self, ensure_copy=False):
        # if ensure_copy:
        return np.frombuffer(self.impl.buffer(), dtype=np.double).copy()
        # else:
        #    return np.frombuffer(self.impl.buffer(), dtype=np.double)

    @property
    def data(self):
        return np.frombuffer(self.impl.buffer(), dtype=np.double)

    def __eq__(self, other):
        return type(self) == type(other) and self.impl == other.impl

    def __getitem__(self, ind):
        return self.impl[ind]

    def __setitem__(self, ind, val):
        self.impl[ind] = val

    def __len__(self):
        return self.dim

    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    @classmethod
    def make_zeros(cls, subtype):
        impl = subtype[0](subtype[1], 0.0)
        return DuneXtLaVector(impl)

    @property
    def subtype(self):
        return (type(self.impl), self.impl.dim)

    def copy(self, deep=False):
        return DuneXtLaVector(self.impl.copy())

    def scal(self, alpha):
        self.impl.scal(alpha)

    def axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def dot(self, other):
        return self.impl.dot(other.impl)

    def inner(self, other, product=None):
        assert product is None, "Not implemented"
        return self.dot(other)

    def norm(self):
        return self.l2_norm()

    def norm2(self):
        return self.l2_norm2()

    def l1_norm(self):
        return self.impl.l1_norm()

    def l2_norm(self):
        return self.impl.l2_norm()

    def l2_norm2(self):
        return self.impl.l2_norm() ** 2

    def sup_norm(self):
        return self.impl.sup_norm()

    def dofs(self, dof_indices):
        # assert 0 <= np.min(dof_indices)
        # assert np.max(dof_indices) < self.dim
        return np.array([self.impl[int(i)] for i in dof_indices])

    def amax(self):
        return self.impl.amax()

    def __add__(self, other):
        return DuneXtLaVector(self.impl + other.impl)

    def __iadd__(self, other):
        self.impl += other.impl
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return DuneXtLaVector(self.impl - other.impl)

    def __isub__(self, other):
        self.impl -= other.impl
        return self

    def __mul__(self, other):
        return DuneXtLaVector(self.impl * other)

    def __neg__(self):
        return DuneXtLaVector(-self.impl)

    def __getstate__(self):
        return type(self.impl), self.data

    def __setstate__(self, state):
        self.impl = state[0](len(state[1]), 0.0)
        self.data[:] = state[1]


class DuneXtLaListVectorSpace(ListVectorSpace):
    def __init__(self, dim, id=None):
        self.dim = dim
        self.id = id

    def __eq__(self, other):
        return type(other) is DuneXtLaListVectorSpace and self.dim == other.dim and self.id == other.id

    @classmethod
    def space_from_vector_obj(cls, vec, id_):
        return cls(len(vec), id_)

    @classmethod
    def space_from_dim(cls, dim, id):
        return cls(dim, id)

    def zero_vector(self):
        return DuneXtLaVector(CommonDenseVector(self.dim, 0))

    def make_vector(self, obj):
        return DuneXtLaVector(obj)

    @classmethod
    def from_memory(cls, numpy_array):
        (num_vecs, dim) = numpy_array.shape
        vecs = []
        for i in range(num_vecs):
            vecs.append(DuneXtLaVector(CommonDenseVector.create_from_buffer(numpy_array.data, i * dim, dim)))
        space = DuneXtLaListVectorSpace(dim)
        return ListVectorArray(vecs, space)

    def vector_from_numpy(self, data, ensure_copy=False):
        # TODO: do not copy if ensure_copy is False
        v = self.zero_vector()
        v.data[:] = data
        return v
