#!/bin/bash

#SBATCH --time=00:15:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1 #--constraint=pascal
#SBATCH --qos=standby
#SBATCH --partition=m13h,m13l,mgh,cssp1,dw,m9g,cs
#SBATCH --mem=16G
#SBATCH -J "testing"   # job name
#SBATCH --mail-user=dhodge@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/cuda/12.5/lib64

source ~/PGD2/bin/activate
module load cuda

python main.py
