#!/bin/bash --login
#SBATCH -p hpcpool        # The "partition" - named hpcpool
#SBATCH -N 4             # (or --nodes=) Minimum is 4, Max is 4. Job uses 32 cores on each node.
#SBATCH -n 128             # (or --ntasks=) TOTAL number of tasks. Maximum for testing is 64 .
#SBATCH -t 2-0       # Maximum wallclock 4-0 (4-days).
#SBATCH -A hpc-am-vdwstructs  # Use your HPC project code


source /mnt/iusers01/fatpou01/phy01/a97824ab/softwares/conda_setup.sh
conda activate ml_june2025

for path in ../Calc_wavecar/mp-*; do
	echo ${path}
  	python main.py "$path" 
done
