import numpy as np
import os
import dill
import time

from pymor.core.base import ImmutableObject
from pymor.algorithms.projection import project
from pymor.operators.constructions import LincombOperator, VectorOperator, IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.operators.block import BlockColumnOperator, BlockOperator
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.residual import ResidualOperator
from pymor.parameters.functionals import (ExpressionParameterFunctional,
                                          MaxThetaParameterFunctional,
                                          ProjectionParameterFunctional)
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parallel.dummy import DummyPool

from gridlod import fem

from pdeopt.discretize_gridlod import GridlodModel
from pdeopt.reductor import NonAssembledRBReductor

from rblod.parameterized_stage_1 import CorrectorProblem_for_all_rhs
from rblod.optimized_rom import OptimizedNumpyModelStage1
from rblod.two_scale_model import Two_Scale_Problem
from rblod.two_scale_reductor import CoerciveRBReductorForTwoScale, TrueDiagonalBlockOperator

from pymor.reductors.coercive_ipl3dg import CoerciveIPLD3GRBReductor
from pymor.reductors.coercive_ipl3dg import project_block_operator, project_block_rhs

class QuadraticPdeoptStationaryCoerciveLRBMSReductor(CoerciveRBReductor):
    def __init__(self, fom, f, dd_grid=None, opt_product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None, unique_basis=False,
                 reductor_type='simple_coercive', mu_bar=None,
                 prepare_for_hessian=False, parameter_space=None,
                 optional_enrichment=False, store_in_tmp=False, use_fine_mesh=True,
                 print_on_ranks=True, pool=None):
        tic = time.perf_counter()
        self.__auto_init(locals())

        # if self.opt_product is None:
        #     self.opt_product = fom.opt_product
        # super().__init__(fom, product=opt_product, check_orthonormality=check_orthonormality,
        #                  check_tol=check_tol, coercivity_estimator=coercivity_estimator)

        # this is for the two scale model. Also needed for RBLOD estimator !!
        self.pool = pool or DummyPool()
        # self.precompute_constants()
        self.initialize_roms()

        del self.pool
        print(f'Initialization took {time.perf_counter() -tic:.4f}s')

    def initialize_roms(self):
        print('initializing roms ...')
        # min_alpha = self.min_alpha
        # coercivity_estimator = lambda mu: min_alpha
        self.primal_reductor = CoerciveIPLD3GRBReductor(self.fom, self.dd_grid)
        self.dual_reductor = CoerciveIPLD3GRBReductor(self.fom, self.dd_grid)
        print('') if self.print_on_ranks else 0

    def reduce(self):
        # NOTE: never use super().reduce() for this since the dims are not correctly computed here !
        return self._reduce()

    def _reduce(self):
        tic = time.perf_counter()
        if self.assemble_error_estimator.__func__ is not ProjectionBasedReductor.assemble_error_estimator:
            with self.logger.block('Assembling error estimator ...'):
                error_estimator = self.assemble_error_estimator()
        else:
            error_estimator = None

        with self.logger.block('Building ROM ...'):
            rom = self.build_rom(error_estimator)
            rom = rom.with_(name=f'{self.fom.name}_reduced')
            rom.disable_logging()
        print(f' ... construction took {time.perf_counter() - tic:.4f}s')
        return rom

    def build_rom(self, estimator):
        print('constructing ROM ...', end='', flush=True)
        evaluation_counter = self.fom.evaluation_counter

        # reduce primal
        self.primal_model = self.primal_reductor.reduce()
        # build dual reductor
        self.dual_model = self.construct_dual_model()

        projected_output = self.project_output()
        projected_product = self.project_product()

        primal_bases_size = self.primal_reductor.basis_length()
        dual_bases_size = self.dual_reductor.basis_length()
        print(f' ... Models have been constructed ... length of bases are '
              f'{primal_bases_size, dual_bases_size}')

        return self.fom.with_(primal_model=self.primal_model, estimators=None,
                              dual_model=self.dual_model, fom=self.fom,
                              evaluation_counter=evaluation_counter,
                              opt_product=projected_product,
                              coarse_projection=projected_product,
                              output_functional_dict=projected_output)

    def construct_dual_model(self):
        # TODO: assertions
        suffix = ''
        bilinear_part = self.fom.output_functional_dict[f'd_u_bilinear_part{suffix}']
        d_u_linear_part = self.fom.output_functional_dict[f'd_u_linear_part{suffix}']
        dual_rhs_operators = np.empty(self.primal_reductor.S, dtype=object)
        for I in range(self.primal_reductor.S):
            rhs_operators = list(d_u_linear_part.blocks[I, 0].operators)
            rhs_coefficients = list(d_u_linear_part.blocks[I, 0].coefficients)
            bilinear_part_block = bilinear_part.blocks[I][I]
            local_basis = self.primal_reductor.local_bases[I]
            for i in range(len(local_basis)):
                u = local_basis[i]
                if isinstance(bilinear_part_block, LincombOperator):
                    for j, op in enumerate(bilinear_part_block.operators):
                        # TODO: THIS IS PROBABLY WRONG AND HAS TO BE APPLIED GLOBALLY
                        rhs_operators.append(VectorOperator(op.apply(u)))
                        rhs_coefficients.append(
                            ProjectionParameterFunctional(f'basis_coefficients_{I}',
                                                          len(local_basis), i)
                            * bilinear_part_block.coefficients[j]
                        )
                else:
                    rhs_operators.append(VectorOperator(bilinear_part_block.apply(u, None)))
                    rhs_coefficients.append(
                        1. * ProjectionParameterFunctional(f'basis_coefficients_{I}',
                                                          len(local_basis), i)
                    )

            dual_rhs_operators[I] = LincombOperator(rhs_operators, rhs_coefficients)

        dual_rhs_operator = BlockColumnOperator(dual_rhs_operators)
        dual_intermediate_fom = self.fom.primal_model.with_(rhs = dual_rhs_operator)

        dual_auxiliary_reductor = CoerciveIPLD3GRBReductor(
            dual_intermediate_fom, self.dd_grid, local_bases=self.dual_reductor.local_bases
        )

        dual_rom = dual_auxiliary_reductor.reduce()
        return dual_rom

    def extend_bases(self, mu, U = None, P = None, corT = None, pool=None):
        print('extending bases...', end='', flush=True)
        tic = time.perf_counter()
        # if pool is None:
        #     print('WARNING: You are not using a parallel pool')
        pool = pool or DummyPool()
        # enrich primal
        self.primal_reductor.enrich_all_locally(mu, use_global_matrix=False)
        # enrich dual
        self.dual_reductor.add_global_solutions(self.fom.solve_dual(mu))
        return U, P

    def add_global_solutions(self, mu):
        U = self.fom.solve(mu)
        self.primal_reductor.add_global_solutions(U)
        self.dual_reductor.add_global_solutions(self.fom.solve_dual(mu, U))

    def assemble_error_estimator(self, RB_primal=None, RB_dual=None):
        estimators = {}

        # TODO: FILL THIS

        ##########################################
        estimators['u_d_mu'] = None
        estimators['p_d_mu'] = None
        estimators['output_functional_hat_d_mus'] = None
        estimators['hessian_d_mu_il'] = None

        return estimators

    def project_output(self):
        output_functional = self.fom.output_functional_dict
        li_part = output_functional['linear_part']
        bi_part = output_functional['bilinear_part']
        d_u_li_part = output_functional['d_u_linear_part']
        d_u_bi_part = output_functional['d_u_bilinear_part']

        projected_li_part = project_block_rhs(li_part, self.primal_reductor.local_bases)
        projected_bi_part = project_block_operator(bi_part, self.primal_reductor.local_bases,
                                                   self.primal_reductor.local_bases)
        projected_d_u_bi_part = project_block_operator(d_u_bi_part, self.primal_reductor.local_bases,
                                                     self.primal_reductor.local_bases)
        projected_d_u_li_part = project_block_rhs(d_u_li_part, self.primal_reductor.local_bases)
        dual_projected_d_u_bi_part = project_block_operator(bi_part, self.dual_reductor.local_bases,
                                                          self.dual_reductor.local_bases)
        dual_primal_projected_primal_operator = project_block_operator(
            self.fom.primal_model.operator, self.dual_reductor.local_bases,
            self.primal_reductor.local_bases
        )
        dual_projected_primal_rhs = project_block_rhs(self.fom.primal_model.rhs,
                                                      self.dual_reductor.local_bases)

        projected_functionals = {
           'output_coefficient' : output_functional['output_coefficient'],
           'linear_part' : projected_li_part,
           'bilinear_part' : projected_bi_part,
           'd_u_linear_part' : projected_d_u_li_part,
           'd_u_bilinear_part' : projected_d_u_bi_part,
           'dual_projected_d_u_bilinear_part' : dual_projected_d_u_bi_part,
           'dual_primal_projected_op': dual_primal_projected_primal_operator,
           'dual_projected_rhs': dual_projected_primal_rhs,
        }
        return projected_functionals

    def project_product(self):
        opt_product = self.fom.opt_product
        projected_product = project_block_operator(opt_product,
                                                   self.primal_reductor.local_bases,
                                                   self.primal_reductor.local_bases)
        return projected_product

    def assemble_estimator_for_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_primal_subbasis(self, dim):
        raise NotImplementedError

