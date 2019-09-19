import libhapodgdt

solver = libhapodgdt.CellModelSolver('single_cell', 20, 5)
filename = 'from_python'
subsampling = True
solver.visualize(filename, 0, 0., subsampling)
t = 0.
t_end = 0.002
dt = 1e-3
write_step = 1e-3
next_save_time = t_end if t + write_step > t_end else t + write_step
save_step_counter = 1

vecs = solver.solve(t_end, dt, write_step, filename, True)

#num_cells = solver.num_cells()
#pfield_vec = solver.pfield_vector()
#ofield_vec = solver.ofield_vector()
#stokes_vec = solver.stokes_vector()
#while t < t_end:
#      max_dt = dt
#      if t + dt > t_end:
#        max_dt = t_end - t
#      actual_dt = min(dt, max_dt)
#
#      print('Current time: {}'.format(t))
#      for kk in range(num_cells):
#        solver.prepare_pfield_operator(dt, kk)
#        pfield_vec = solver.solve_pfield(pfield_vec, kk)
#        solver.set_pfield_variables(kk, pfield_vec)
#        print('Pfield {} done'.format(kk))
#        solver.prepare_ofield_operator(dt, kk)
#        ofield_vec = solver.solve_ofield(ofield_vec, kk)
#        solver.set_ofield_variables(kk, ofield_vec)
#        print('Ofield {} done'.format(kk))
#
#      solver.prepare_stokes_operator()
#      stokes_vec = solver.solve_stokes()
#      solver.set_stokes_variables(stokes_vec)
#      print('Stokes done')
#
#      t += actual_dt;
#
#      if write_step < 0. or t >= next_save_time:
#        solver.visualize(filename, save_step_counter, t, subsampling)
#        next_save_time += write_step
#        save_step_counter += 1
