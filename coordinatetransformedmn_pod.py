import math
import resource
import time
import sys
from mpi4py import MPI
from timeit import default_timer as timer

import numpy as np
from pymor.algorithms.pod import pod

from hapod.coordinatetransformedmn.utility import create_and_scatter_parameters, convert_L2_l2, calculate_mean_errors, create_coordinatetransformedmn_solver
from hapod.mpi import MPIWrapper

import matplotlib.pyplot as plt

# The POD only uses MPI rank 0, all other processes are busy-waiting for it to finish.
# With idle_wait, the waiting threads do not cause such a high CPU usage.
# Adapted from https://gist.github.com/donkirkby/16a89d276e46abb0a106
def idle_wait(mpi, root):
    # Set this to 0 for maximum responsiveness, but that will peg CPU to 100%
    if mpi.rank_world == root:
        for rank in range(1, mpi.size_world):
            mpi.comm_world.send(0, dest=rank, tag=rank)
    else:
        sleep_seconds = 1
        if sleep_seconds > 0:
            while not mpi.comm_world.Iprobe(source=MPI.ANY_SOURCE):
                # print(f"{mpi.rank_world} waited another second")
                time.sleep(sleep_seconds)
        mpi.comm_world.recv(source=root, tag=mpi.rank_world)

def coordinatetransformedmn_pod(grid_size, l2_tol, testcase, logfile=None):

    # get MPI communicators
    mpi = MPIWrapper()

    # get boltzmann solver to create snapshots
    min_param = 1
    max_param = 8
    mu = create_and_scatter_parameters(testcase, mpi.comm_world, min_param=min_param, max_param=max_param)
    solver = create_coordinatetransformedmn_solver(grid_size, mu, testcase)

    # calculate problem trajectory
    start = timer()
    times, snapshots = solver.solve()
    times = np.array(times)
    mpi.comm_world.Barrier()
    elapsed_data_gen = timer() - start
    num_snapshots = len(snapshots)
    assert len(times) == num_snapshots

    # reduce number of snapshots by not keeping all of the snapshots for locally small timesteps
    t_end = 2.5 if "SourceBeam" in testcase else 3.2
    dx = 3 / grid_size if "SourceBeam" in testcase else 7 / grid_size
    max_sigma_t = 2 * max_param
    dimDomain = 1 if "SourceBeam" in testcase else 3
    dt_nonadaptive = 1. / (math.sqrt(dimDomain) / dx + max_sigma_t)
    print(dt_nonadaptive)

    # intervals adapted to predicted timesteps
    predicted_dt = lambda x: max(3e-11, dt_nonadaptive * (2**(math.tanh(10 * x)) - 1))
    dt_threshold = lambda x: predicted_dt(x) / 10
    t = 0
    predicted_times = [t]
    while t < t_end:
        # t += predicted_dt(t)
        t += dt_threshold(t)
        if t > t_end:
            t = t_end
        predicted_times.append(t)
    max_snaps_per_interval = 50
    num_time_intervals = len(predicted_times) // max_snaps_per_interval

    # fixed length intervals
    # dt_threshold = dt_nonadaptive / 10
    # desired_num_snaps = math.ceil(t_end/dt_threshold)
    # num_time_intervals = desired_num_snaps // max_snaps_per_interval
    # interval_length = t_end / num_time_intervals

    snapshots_np = snapshots.to_numpy()
    selected_snapshots = solver.solution_space.empty()
    selected_times = []
    for interval_index in range(num_time_intervals):

        t_min = predicted_times[interval_index * max_snaps_per_interval]
        # the + (i == num_time_interval - 1) ensures we do not miss snapshots in the last interval
        t_max = predicted_times[(interval_index + 1) *
                                max_snaps_per_interval] if interval_index != num_time_intervals - 1 else t_end + 1

        # fixed length intervals
        # t_min = interval_index * interval_length
        # t_max = (interval_index+1) * interval_length + (interval_index == num_time_intervals - 1)

        indices = np.where(np.logical_and(times >= t_min, times < t_max))
        if len(indices[0]) == 0:
            continue
        times_in_interval = times[indices]
        dts_in_interval = (times_in_interval[1:] - times_in_interval[:-1]).tolist()
        next_index = indices[0][-1] + 1
        next_time = times[next_index] if next_index < len(snapshots) else t_end
        dts_in_interval.append(next_time - times_in_interval[-1])
        snapshots_in_interval = snapshots_np[indices]
        print(dts_in_interval)
        if len(times_in_interval) > max_snaps_per_interval:
            selected_snapshots_in_interval = []
            remaining_local_indices = []
            for j, index in enumerate(indices[0]):
                dt = dts_in_interval[j]
                if dt > dt_threshold(times_in_interval[j]):
                    selected_snapshots_in_interval.append(snapshots_np[index])
                    selected_times.append(times[index])
                else:
                    remaining_local_indices.append(j)
            num_snapshots_missing = max_snaps_per_interval - len(selected_snapshots_in_interval)
            if num_snapshots_missing > 0:
                times_in_interval = times_in_interval[remaining_local_indices]
                snapshots_in_interval = snapshots_in_interval[remaining_local_indices]

                # # include remaining snapshots with largest dt
                # dts_in_interval = [dts_in_interval[i] for i in remaining_local_indices]
                # ordering = np.array(dts_in_interval).argsort()
                # snapshots_in_interval = snapshots_in_interval[ordering[-num_snapshots_missing:]]
                # times_in_interval = times_in_interval[ordering[-num_snapshots_missing:]]
                # selected_times.extend(times_in_interval.tolist())

                # include every nth remaining snapshot
                num_snapshots_left = len(snapshots_in_interval)
                step = num_snapshots_left / num_snapshots_missing
                every_nth_index = [math.floor(i * step) for i in range(num_snapshots_missing)]
                snapshots_in_interval = snapshots_in_interval[every_nth_index]
                times_in_interval = times_in_interval[every_nth_index]
                selected_times.extend(times_in_interval.tolist())

            snapshots_in_interval = np.append(snapshots_in_interval, np.array(selected_snapshots_in_interval), axis=0)
        else:
            selected_times.extend(times_in_interval)
        selected_snapshots.append(solver.solution_space.from_numpy(snapshots_in_interval))

    # # visualize selected times
    # print(selected_times)
    # selected_times.sort()
    # plt.plot(times, [0] * len(times), 'ro')
    # plt.plot(selected_times, [0] * len(selected_times), 'bo')
    # plt.show()

    print(len(selected_snapshots))
    del solver

    # gather snapshots on rank 0
    snapshots, _, total_num_snapshots, _ = mpi.comm_world.gather_on_rank_0(
        snapshots, num_snapshots, num_modes_equal=False)
    selected_snapshots, _, total_num_selected_snapshots, _ = mpi.comm_world.gather_on_rank_0(
        selected_snapshots, len(selected_snapshots), num_modes_equal=False)

    # perform a POD
    elapsed_pod = 0
    svals = None
    if mpi.rank_world == 0:
        snapshots, svals = pod(snapshots, atol=0., rtol=0., l2_err=l2_tol * np.sqrt(total_num_snapshots))
        selected_snapshots, selected_svals = pod(
            selected_snapshots, atol=0., rtol=0., l2_err=l2_tol * np.sqrt(total_num_snapshots))
        with open(f"{testcase}_grid_{grid_size}_svals.log", "w") as svalfile:
            for sval in svals:
                svalfile.write(f"{sval:.15e}\n")
        with open(f"{testcase}_grid_{grid_size}_svals_selected.log", "w") as svalfile:
            for sval in selected_svals:
                svalfile.write(f"{sval:.15e}\n")
        elapsed_pod = timer() - start

        # write statistics to file
        if logfile is not None:
            logfile.write("After the POD, there are " + str(len(snapshots)) + " modes of " + str(total_num_snapshots) +
                          " snapshots left!\n")
            logfile.write("After the POD with selected snapshots, there are " + str(len(selected_snapshots)) +
                          " modes of " + str(total_num_selected_snapshots) + " snapshots left!\n")
            logfile.write("The maximum amount of memory used on rank 0 was: " +
                          str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.**2) + " GB\n")
            elapsed = timer() - start
            logfile.write("Time elapsed: " + str(elapsed) + "\n")
    idle_wait(mpi, root=0)

    return snapshots, selected_snapshots, svals, total_num_snapshots, mu, mpi, elapsed_data_gen, elapsed_pod


if __name__ == "__main__":
    argc = len(sys.argv)
    grid_size = 21 if argc < 2 else int(sys.argv[1])
    L2_tol = 1e-3 if argc < 3 else float(sys.argv[2])
    testcase = "HFM50SourceBeam" if argc < 4 else sys.argv[3]
    filename = f"{testcase}_POD_gridsize_{grid_size}_tol_{L2_tol}.log"
    logfile = open(filename, "a")
    final_modes, final_modes_selected, _, total_num_snapshots, mu, mpi, _, _ = coordinatetransformedmn_pod(
        grid_size, convert_L2_l2(L2_tol, grid_size, testcase), testcase, logfile=logfile)
    final_modes, win = mpi.shared_memory_bcast_modes(final_modes, returnlistvectorarray=True)
    final_modes_selected, win_selected = mpi.shared_memory_bcast_modes(
        final_modes_selected, returnlistvectorarray=True)
    print("Shared broadcast done")
    calculate_mean_errors(final_modes, grid_size, mu, testcase, total_num_snapshots, mpi, logfile=logfile)
    calculate_mean_errors(
        final_modes_selected, grid_size, mu, testcase, total_num_snapshots, mpi, logfile=logfile, selected=True)
    win.Free()
    win_selected.Free()
    logfile.close()
    if mpi.rank_world == 0:
        logfile = open(filename, "r")
        print("\n\n\nResults:\n")
        print(logfile.read())
        logfile.close()
