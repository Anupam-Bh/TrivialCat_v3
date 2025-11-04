#!/bin/bash 
#SBATCH --job-name=SmAlGe
#SBATCH -o stdout.%j 
#SBATCH -e stderr.%j
#SBATCH --time=48:00:00
#SBATCH --ntasks=48

# executable 
#sleep 5m

source ~/softwares/intel/OneAPI_HPC_toolkit/setvars.sh

cd scf/
mpirun -np 24 /home/r730/softwares/vasp_6.4.3_wannier31/bin/vasp_std > output
cp CHGCAR ../bands/
cd ../bands/
mpirun -np 24 /home/r730/softwares/vasp_6.4.3_wannier31/bin/vasp_std > output


#mpirun -np 24 /home/r730/softwares/wannier90-3.1.0/wannier90.x wannier90 
#mpirun -np 24 /home/r730/softwares/wannier_tools-master/bin/wt2.x < wt.in 
