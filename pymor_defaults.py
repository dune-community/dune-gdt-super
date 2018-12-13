# pyMOR defaults config file
# This file has been automatically created by pymor.core.defaults.write_defaults_to_file'.

d = {}

########################################################################
#                                                                      #
# SETTING THE FOLLOWING DEFAULTS WILL AFFECT STATE ID CALCULATION.     #
#                                                                      #
########################################################################

# d['pymor.algorithms.basic.almost_equal.atol'                                   ] = 1e-14
# d['pymor.algorithms.basic.almost_equal.rtol'                                   ] = 1e-14

# d['pymor.algorithms.genericsolvers.options.default_least_squares_solver'       ] = 'least_squares_generic_lsmr'
# d['pymor.algorithms.genericsolvers.options.default_solver'                     ] = 'generic_lgmres'
# d['pymor.algorithms.genericsolvers.options.generic_lgmres_inner_m'             ] = 39
# d['pymor.algorithms.genericsolvers.options.generic_lgmres_maxiter'             ] = 1000
# d['pymor.algorithms.genericsolvers.options.generic_lgmres_outer_k'             ] = 3
# d['pymor.algorithms.genericsolvers.options.generic_lgmres_tol'                 ] = 1e-05
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_atol'    ] = 1e-06
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_btol'    ] = 1e-06
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_conlim'  ] = 100000000.0
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_damp'    ] = 0.0
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_maxiter' ] = None
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsqr_atol'    ] = 1e-06
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsqr_btol'    ] = 1e-06
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsqr_conlim'  ] = 100000000.0
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsqr_iter_lim'] = None

# d['pymor.algorithms.gram_schmidt.gram_schmidt.atol'                            ] = 1e-13
# d['pymor.algorithms.gram_schmidt.gram_schmidt.check'                           ] = True
# d['pymor.algorithms.gram_schmidt.gram_schmidt.check_tol'                       ] = 0.001
# d['pymor.algorithms.gram_schmidt.gram_schmidt.reiterate'                       ] = True
# d['pymor.algorithms.gram_schmidt.gram_schmidt.reiteration_threshold'           ] = 0.1
# d['pymor.algorithms.gram_schmidt.gram_schmidt.rtol'                            ] = 1e-13

# d['pymor.algorithms.newton.newton.atol'                                        ] = -1.0
# d['pymor.algorithms.newton.newton.maxiter'                                     ] = 100
# d['pymor.algorithms.newton.newton.miniter'                                     ] = 0
# d['pymor.algorithms.newton.newton.rtol'                                        ] = -1.0
# d['pymor.algorithms.newton.newton.stagnation_threshold'                        ] = 0.9
# d['pymor.algorithms.newton.newton.stagnation_window'                           ] = 3

# d['pymor.algorithms.pod.pod.atol'                                              ] = 0.0
# d['pymor.algorithms.pod.pod.check'                                             ] = True
# d['pymor.algorithms.pod.pod.check_tol'                                         ] = 1e-10
# d['pymor.algorithms.pod.pod.l2_mean_err'                                       ] = 0.0
# d['pymor.algorithms.pod.pod.orthonormalize'                                    ] = True
# d['pymor.algorithms.pod.pod.rtol'                                              ] = 4e-08
# d['pymor.algorithms.pod.pod.symmetrize'                                        ] = False

# d['pymor.operators.constructions.induced_norm.raise_negative'                  ] = True
# d['pymor.operators.constructions.induced_norm.tol'                             ] = 1e-10

# d['pymor.operators.fv.jacobian_options.delta'                                  ] = 1e-07

# d['pymor.operators.numpy.dense_options.default_least_squares_solver'           ] = 'least_squares_lstsq'
# d['pymor.operators.numpy.dense_options.default_solver'                         ] = 'solve'
# d['pymor.operators.numpy.dense_options.least_squares_lstsq_rcond'              ] = -1.0

# d['pymor.operators.numpy.sparse_options.bicgstab_maxiter'                      ] = None
# d['pymor.operators.numpy.sparse_options.bicgstab_tol'                          ] = 1e-15
# d['pymor.operators.numpy.sparse_options.default_least_squares_solver'          ] = 'least_squares_lsmr'
# d['pymor.operators.numpy.sparse_options.default_solver'                        ] = 'spsolve'
# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_atol'               ] = 1e-06
# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_btol'               ] = 1e-06
# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_conlim'             ] = 100000000.0
# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_damp'               ] = 0.0
# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_maxiter'            ] = None
# d['pymor.operators.numpy.sparse_options.least_squares_lsqr_atol'               ] = 1e-06
# d['pymor.operators.numpy.sparse_options.least_squares_lsqr_btol'               ] = 1e-06
# d['pymor.operators.numpy.sparse_options.least_squares_lsqr_conlim'             ] = 100000000.0
# d['pymor.operators.numpy.sparse_options.least_squares_lsqr_iter_lim'           ] = None
# d['pymor.operators.numpy.sparse_options.lgmres_inner_m'                        ] = 39
# d['pymor.operators.numpy.sparse_options.lgmres_maxiter'                        ] = 1000
# d['pymor.operators.numpy.sparse_options.lgmres_outer_k'                        ] = 3
# d['pymor.operators.numpy.sparse_options.lgmres_tol'                            ] = 1e-05
# d['pymor.operators.numpy.sparse_options.pyamg_maxiter'                         ] = 400
# d['pymor.operators.numpy.sparse_options.pyamg_rs_CF'                           ] = 'RS'
# d['pymor.operators.numpy.sparse_options.pyamg_rs_accel'                        ] = None
# d['pymor.operators.numpy.sparse_options.pyamg_rs_coarse_solver'                ] = 'pinv2'
# d['pymor.operators.numpy.sparse_options.pyamg_rs_cycle'                        ] = 'V'
# d['pymor.operators.numpy.sparse_options.pyamg_rs_max_coarse'                   ] = 500
# d['pymor.operators.numpy.sparse_options.pyamg_rs_max_levels'                   ] = 10
# d['pymor.operators.numpy.sparse_options.pyamg_rs_maxiter'                      ] = 100
# d['pymor.operators.numpy.sparse_options.pyamg_rs_postsmoother'                 ] = ('gauss_seidel', {'sweep': 'symmetric'})
# d['pymor.operators.numpy.sparse_options.pyamg_rs_strength'                     ] = ('classical', {'theta': 0.25})
# d['pymor.operators.numpy.sparse_options.pyamg_rs_tol'                          ] = 1e-05
# d['pymor.operators.numpy.sparse_options.pyamg_sa_accel'                        ] = None
# d['pymor.operators.numpy.sparse_options.pyamg_sa_aggregate'                    ] = 'standard'
# d['pymor.operators.numpy.sparse_options.pyamg_sa_coarse_solver'                ] = 'pinv2'
# d['pymor.operators.numpy.sparse_options.pyamg_sa_cycle'                        ] = 'V'
# d['pymor.operators.numpy.sparse_options.pyamg_sa_diagonal_dominance'           ] = False
# d['pymor.operators.numpy.sparse_options.pyamg_sa_improve_candidates'           ] = [('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None]
# d['pymor.operators.numpy.sparse_options.pyamg_sa_max_coarse'                   ] = 500
# d['pymor.operators.numpy.sparse_options.pyamg_sa_max_levels'                   ] = 10
# d['pymor.operators.numpy.sparse_options.pyamg_sa_maxiter'                      ] = 100
# d['pymor.operators.numpy.sparse_options.pyamg_sa_postsmoother'                 ] = ('block_gauss_seidel', {'sweep': 'symmetric'})
# d['pymor.operators.numpy.sparse_options.pyamg_sa_presmoother'                  ] = ('block_gauss_seidel', {'sweep': 'symmetric'})
# d['pymor.operators.numpy.sparse_options.pyamg_sa_smooth'                       ] = ('jacobi', {'omega': 1.3333333333333333})
# d['pymor.operators.numpy.sparse_options.pyamg_sa_strength'                     ] = 'symmetric'
# d['pymor.operators.numpy.sparse_options.pyamg_sa_symmetry'                     ] = 'hermitian'
# d['pymor.operators.numpy.sparse_options.pyamg_sa_tol'                          ] = 1e-05
# d['pymor.operators.numpy.sparse_options.pyamg_tol'                             ] = 1e-05
# d['pymor.operators.numpy.sparse_options.spilu_drop_rule'                       ] = 'basic,area'
# d['pymor.operators.numpy.sparse_options.spilu_drop_tol'                        ] = 0.0001
# d['pymor.operators.numpy.sparse_options.spilu_fill_factor'                     ] = 10
# d['pymor.operators.numpy.sparse_options.spilu_permc_spec'                      ] = 'COLAMD'
# d['pymor.operators.numpy.sparse_options.spsolve_keep_factorization'            ] = True
# d['pymor.operators.numpy.sparse_options.spsolve_permc_spec'                    ] = 'COLAMD'

# d['pymor.parallel.default.new_parallel_pool.allow_mpi'                         ] = False
# d['pymor.parallel.default.new_parallel_pool.ipython_num_engines'               ] = None
# d['pymor.parallel.default.new_parallel_pool.ipython_profile'                   ] = None

# d['pymor.tools.floatcmp.float_cmp.atol'                                        ] = 1e-14
# d['pymor.tools.floatcmp.float_cmp.rtol'                                        ] = 1e-14

d['pymor.tools.mpi.event_loop_settings.auto_launch'                            ] = False

# d['pymor.tools.random.new_random_state.seed'                                   ] = 42


########################################################################
#                                                                      #
# SETTING THE FOLLOWING DEFAULTS WILL NOT AFFECT STATE ID CALCULATION. #
#                                                                      #
########################################################################

# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsmr_show'    ] = False
# d['pymor.algorithms.genericsolvers.options.least_squares_generic_lsqr_show'    ] = False

# d['pymor.core.cache.default_regions.disk_max_size'                             ] = 1073741824
# d['pymor.core.cache.default_regions.disk_path'                                 ] = '/tmp/pymor.cache.tobias'
# d['pymor.core.cache.default_regions.memory_max_keys'                           ] = 1000
# d['pymor.core.cache.default_regions.persistent_max_size'                       ] = 1073741824
# d['pymor.core.cache.default_regions.persistent_path'                           ] = '/tmp/pymor.persistent.cache.tobias'

# d['pymor.core.logger.getLogger.filename'                                       ] = ''

# d['pymor.core.logger.set_log_format.block_timings'                             ] = False
# d['pymor.core.logger.set_log_format.indent_blocks'                             ] = True
# d['pymor.core.logger.set_log_format.max_hierarchy_level'                       ] = 1

# d['pymor.core.logger.set_log_levels.levels'                                    ] = {'pymor': 'INFO'}

# d['pymor.gui.qt.visualize_patch.backend'                                       ] = 'gl'

# d['pymor.operators.numpy.sparse_options.least_squares_lsmr_show'               ] = False
# d['pymor.operators.numpy.sparse_options.least_squares_lsqr_show'               ] = False
# d['pymor.operators.numpy.sparse_options.pyamg_verb'                            ] = False

# d['pymor.playground.vectorarrays.disk.basedir.path'                            ] = '/tmp/pymor.diskarray.tobias'

# d['pymor.tools.pprint.format_array.compact_print'                              ] = False


