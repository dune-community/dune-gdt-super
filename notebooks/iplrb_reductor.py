import numpy as np

from pymor.operators.constructions import ZeroOperator, LincombOperator, VectorOperator
from pymor.algorithms.projection import project
from pymor.operators.block import BlockOperator

from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.gram_schmidt import gram_schmidt

class EllipticIPDGReductor(CoerciveRBReductor):
    def __init__(self, fom):
        self.S = fom.solution_space.empty().num_blocks
        self.fom = fom

        self.local_bases = [fom.solution_space.empty().block(ss).empty()
                            for ss in range(self.S)]

    def add_global_solutions(self, us):
        assert us in self.fom.solution_space
        for ss in range(self.S):
            us_block = us.block(ss)
            self.local_bases[ss].append(us_block)
            # TODO: add offset
            self.local_bases[ss] = gram_schmidt(self.local_bases[ss])

    def add_local_solutions(self, ss, u):
        self.local_bases[ss].append(u)
        # TODO: add offset
        self.local_bases[ss] = gram_schmidt(self.local_bases[ss])

    def basis_length(self):
        return [len(self.local_bases[ss]) for ss in range(self.S)]

    def reduce(self):
        return self._reduce()

    def project_operators(self):
        projected_ops_blocks = []
        # this is for BlockOperator(LincombOperators)
        projected_ops = np.empty((self.S, self.S), dtype=object)
        for ss in range(self.S):
            for nn in range(self.S):
                local_basis_ss = self.local_bases[ss]
                local_basis_nn = self.local_bases[nn]
                if self.fom.operator.blocks[ss][nn]:
                    projected_ops[ss][nn] = project(self.fom.operator.blocks[ss][nn],
                                                    local_basis_ss, local_basis_nn)
        projected_operator = BlockOperator(projected_ops)

        rhs = np.empty(self.S, dtype=object)
        for ss in range(self.S):
            local_basis_ss = self.local_bases[ss]
            rhs_vector = VectorOperator(self.fom.rhs.array.block(ss))
            rhs_int = project(rhs_vector, local_basis_ss, None).matrix[:,0]
            rhs[ss] = projected_ops[ss][ss].range.make_array(rhs_int)
        projected_rhs = VectorOperator(projected_operator.range.make_array(rhs))

        projected_operators = {
            'operator':          projected_operator,
            'rhs':               projected_rhs,
            'products':          None,
            'output_functional': None
        }
        return projected_operators

    def assemble_error_estimator(self):
        return None

    def reconstruct(self, u_rom):
        u_ = []
        for ss in range(self.S):
            basis = self.local_bases[ss]
            u_ss = u_rom.block(ss)
            u_.append(basis.lincomb(u_ss.to_numpy()))
        return self.fom.solution_space.make_array(u_)
