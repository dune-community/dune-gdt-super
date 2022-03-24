import os

filetemplate = r"""#!/bin/bash

#SBATCH --nodes=4                              # set the number of nodes
#SBATCH --ntasks-per-node=36                   # use 36 MPI process per node (no hyperthreads)
#SBATCH --cpus-per-task=1                      # use 1 core per MPI process (no thread-parallel computing)
#SBATCH --exclusive                            # use full node
#SBATCH --mem=80GB                             # How much memory is needed (per node)
#SBATCH --partition=normal                     # set a partition
#SBATCH --time=JOB_TIME                        # set max wallclock time
#SBATCH --job-name=JOB_NAME                    # set name of job
#SBATCH --mail-type=ALL                        # mail alert at start, end and abortion of execution
#SBATCH --mail-user=l_tobi01@wwu.de            # send mail to this address
#SBATCH --output=/scratch/tmp/l_tobi01/job_output/OUTPUT_FILENAME.dat  # set an output file


module load palma/2021a
module load intel Eigen Python CMake Ninja
module li

# Only if you are using Intel MPI (as we do here) - not needed for OpenMPI
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

# run the application
cd ~/hapod-dune-gdt-super
source /scratch/tmp/l_tobi01/icc/dune-python-env/bin/activate
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 srun -n $SLURM_NTASKS --mpi=pmi2 --export=ALL python3 cellmodel_hapod_deim.py cell_isolation_experiment 0.5 0.001 GRID_SIZE GRID_SIZE PFIELD_TOL OFIELD_TOL STOKES_TOL PFIELD_DEIM_TOL OFIELD_DEIM_TOL STOKES_DEIM_TOL True log_and_log_inverted method_of_snapshots
"""

job_time = "1-12:00:00"
grid_size = 40
stokes_tol = 0
ofield_tol = 0
pfield_tol = 0
pfield_deim_tol = 0
ofield_deim_tol = 0
stokes_deim_tol = 0
############ Choose one of the following values for landscape ################
# landscape = "pfield"
# landscape = "ofield"
# landscape = "stokes"
landscape = "all_same_tol"
# landscape = "all_stokes_higher_tol"
#############################################################################
tmp_dir = "generated_submit_files"
if not os.path.exists(tmp_dir):
   os.mkdir(tmp_dir)
for tol in (1e-01, 1e-02, 1e-03, 1e-04, 1e-05):
    for deim_tol in (1e-06, 1e-07, 1e-08, 1e-09, 1e-10):
        if landscape == "pfield":
            pfield_tol = tol
            pfield_deim_tol = deim_tol
        elif landscape == "ofield":
            ofield_tol = tol
            ofield_deim_tol = deim_tol
        elif landscape == "stokes":
            stokes_tol = tol
            stokes_deim_tol = deim_tol
        elif landscape == "all_same_tol":
            pfield_tol = ofield_tol = stokes_tol = tol
            pfield_deim_tol = ofield_deim_tol = stokes_deim_tol = deim_tol
        elif landscape == "all_stokes_higher_tol":
            pfield_tol = ofield_tol = tol
            stokes_tol = tol * 10
            pfield_deim_tol = ofield_deim_tol = deim_tol
            stokes_deim_tol = deim_tol * 10
        else:
            raise NotImplementedError("Unknown value for landscape")
        job_name = f"hapod2_grid{grid_size}_1tppr_tend0.5_dt0.001_{pfield_tol}_{ofield_tol}_{stokes_tol}_{pfield_deim_tol}_{ofield_deim_tol}_{stokes_deim_tol}"
        output_filename = f"{job_name}_output"
        submit_filename = f"{tmp_dir}/{job_name}.submit"
        filecontents = (
            filetemplate.replace("JOB_TIME", job_time)
            .replace("JOB_NAME", job_name)
            .replace("OUTPUT_FILENAME", output_filename)
            .replace("GRID_SIZE", str(grid_size))
            .replace("PFIELD_TOL", str(pfield_tol))
            .replace("OFIELD_TOL", str(ofield_tol))
            .replace("STOKES_TOL", str(stokes_tol))
            .replace("PFIELD_DEIM_TOL", str(pfield_deim_tol))
            .replace("OFIELD_DEIM_TOL", str(ofield_deim_tol))
            .replace("STOKES_DEIM_TOL", str(stokes_deim_tol))
        )
        with open(submit_filename, "w") as f:
            f.writelines(filecontents)
        os.system(f"sbatch {submit_filename}")

