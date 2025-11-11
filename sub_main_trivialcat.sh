#!/bin/bash 
#SBATCH --job-name=jobname
#SBATCH -o stdout.%j 
#SBATCH -e stderr.%j
#SBATCH --time=48:00:00
#SBATCH --ntasks=48


#source /mnt/iusers01/fatpou01/phy01/a97824ab/softwares/conda_setup.sh
source ~/software/conda_setup.sh 
conda activate trivialcat


for path in Calc_wavecar_new/mp-*; do
	echo ${path}
  	python main_trivialcat.py "$path" 
done
