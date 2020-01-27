import sys
import numpy as np
from hapod.cellmodel.wrapper import CellModelSolver, CellModel

from pymor.core.defaults import set_defaults
from pymor.core.logger import set_log_levels

set_defaults({'pymor.algorithms.newton.newton.atol':1e-12, 'pymor.algorithms.newton.newton.rtol':1e-13})
# set_log_levels({'pymor.algorithms.newton': 'WARN'})

# Checks if python results for several mu still match c++ results
def solve_test():
    mu = {'Pa': 1., 'Be': 0.3, 'Ca': 0.1}
    solver = CellModelSolver('single_cell', 2e-3, 20, 5, mu)
    m = CellModel(solver, 1e-3, 2e-3)
    expected_norms = [[[1705.22577705, 14160.65288329, 14820.49753363], [6.74784164, 40.19907385, 40.69457881],
                       [1., 303.91427767, 314.96366696]],
                      [[1705.22577705, 13662.82221481, 14291.88059816], [6.74784164, 39.41113454, 39.8901289],
                       [1., 290.71846422, 301.31239531]],
                      [[1705.22577705, 4280.78359353, 4285.51273652], [6.74784164, 17.69810992, 17.6292538],
                       [1., 45.3560837, 45.33209378]],
                      [[1705.22577705, 3873.36072581, 3868.4561933], [6.74784164, 16.08919339, 15.9980123],
                       [1., 32.60116473, 32.54643408]],
                      [[1705.22577705, 13854.61787406, 14485.83477615], [6.74784164, 39.88651202, 40.28830549],
                       [1., 325.11656523, 314.45107771]],
                      [[1705.22577705, 13357.03762648, 13959.79699229], [6.74784164, 39.08354243, 39.46985783],
                       [1., 311.94618705, 300.73566054]],
                      [[1705.22577705, 3711.32217309, 3733.73682514], [6.74784164, 15.96200882, 15.9560587],
                       [1., 50.66500226, 49.56465551]],
                      [[1705.22577705, 3137.35880294, 3152.76229929], [6.74784164, 13.53489397, 13.53888896],
                       [1., 34.43501477, 33.98236335]]]
    i = 0
    for Pa in [0.1, 1]:
        for Be in [0.3, 3]:
            for Ca in [0.1, 1]:
                # U = m.solve_and_check(mu={'Pa': Pa, 'Be': Be, 'Ca': Ca})
                U = m.solve(mu={'Pa': Pa, 'Be': Be, 'Ca': Ca})
                norms = [list(U.l1_norm()), list(U.l2_norm()), list(U.sup_norm())]
                if not np.allclose(norms, expected_norms[i]):
                    print("Results changed for parameter ({}, {}, {})".format(Be, Ca, Pa))
                    print(norms, " vs. ", expected_norms[i])
                #print("[{}, {}, {}]".format(U.l1_norm(), U.l2_norm(), U.sup_norm()))
                m.visualize(U, prefix="py_Be{}Ca{}Pa{}".format(Be, Ca, Pa), subsampling=True)
                i += 1


if __name__ == "__main__":
    solve_test()
