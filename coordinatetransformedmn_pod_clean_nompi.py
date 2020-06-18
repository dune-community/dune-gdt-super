import sys
import numpy as np
from pymor.algorithms.pod import pod

from hapod.coordinatetransformedmn.utility import (
    create_parameters,
    convert_L2_l2,
    calculate_mean_errors,
    create_coordinatetransformedmn_solver,
)


def coordinatetransformedmn_pod(mu_count, grid_size, l2_tol, testcase, logfile=None):

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mus = create_parameters(testcase, mu_count, min_param=min_param, max_param=max_param)

    all_snapshots = None

    for mu in mus:
        solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)

        if all_snapshots is None:
            all_snapshots = solver.solution_space.empty()
            all_nonlinear_snapshots = solver.solution_space.empty()

        # calculate problem trajectory
        times, snapshots, nonlinear_snapshots = solver.solve(store_operator_evaluations=True)
        num_snapshots = len(snapshots)
        assert len(times) == num_snapshots

        all_snapshots.append(snapshots, remove_from_other=True)
        all_nonlinear_snapshots.append(nonlinear_snapshots, remove_from_other=True)
        del solver
        print('******', len(all_snapshots))

    basis, svals = pod(all_snapshots, atol=0.0, rtol=0.0, l2_err=l2_tol * np.sqrt(len(all_snapshots)))
    if logfile is not None:
        logfile.write("After the POD, there are " + str(len(basis)) + " modes of " + str(len(all_snapshots)) + " snapshots left!\n")

    return basis, svals, all_snapshots, mus


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 100 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-1 if argc < 3 else float(sys.argv[2])
    testcase = "HFM50SourceBeam" if argc < 4 else sys.argv[3]
    filename = f"{testcase}_POD_gridsize_{grid_size}_tol_{L2_tol}.log"
    logfile = open(filename, "a")
    basis, _, all_snapshots, mus = coordinatetransformedmn_pod(
        1, grid_size, convert_L2_l2(L2_tol, grid_size, testcase), testcase, logfile=logfile
    )
    err = convert_L2_l2(np.linalg.norm((all_snapshots - basis.lincomb(all_snapshots.dot(basis))).norm()) / np.sqrt(len(all_snapshots)),
                        grid_size, testcase, input_is_l2=True)
    logfile.write(f'Mean L2-err: {err}\n')
    logfile.close()
    logfile = open(filename, "r")
    print("\n\n\nResults:\n")
    print(logfile.read())
    logfile.close()
