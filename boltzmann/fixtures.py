from itertools import product

import numpy as np

from boltzmann.wrapper import IMPL_TYPES, DuneStuffVectorSpace, DuneStuffVector


from pymortests.fixtures.numpy import (numpy_vector_array_factory,
                                       numpy_vector_array_factory_arguments,
                                       numpy_vector_array_factory_arguments_pairs_with_same_dim,
                                       numpy_vector_array_factory_arguments_pairs_with_different_dim)


def dune_stuff_vector_array_factory(impl_type, length, dim, seed):
    array = numpy_vector_array_factory(length, dim, seed).data
    U = DuneStuffVectorSpace(impl_type, dim).zeros(length)
    for u, x in zip(U._list, array):
        u.data[:] = x
    return U


dune_stuff_vector_array_generators = \
    [lambda args=args: dune_stuff_vector_array_factory(args[0], *(args[1]))
     for args in product(IMPL_TYPES, numpy_vector_array_factory_arguments)]


dune_stuff_vector_array_pair_with_same_dim_generators = \
    [lambda t=t, l=l, l2=l2, d=d, s1=s1, s2=s2: (dune_stuff_vector_array_factory(t, l, d, s1),
                                                 dune_stuff_vector_array_factory(t, l2, d, s2))
     for t, (l, l2, d, s1, s2) in product(IMPL_TYPES, numpy_vector_array_factory_arguments_pairs_with_same_dim)]


dune_stuff_vector_array_pair_with_different_dim_generators = \
    [lambda t=t, l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (dune_stuff_vector_array_factory(t, l, d1, s1),
                                                          dune_stuff_vector_array_factory(t, l2, d2, s2))
     for t, (l, l2, d1, d2, s1, s2) in product(IMPL_TYPES, numpy_vector_array_factory_arguments_pairs_with_different_dim)]


def equality_test(first, second):
    return (first - second).sup_norm() == 0.

from pymortests.pickling import is_equal_dispatch_table
is_equal_dispatch_table[DuneStuffVector] = equality_test
