#!/bin/bash  
#SBATCH -J qutip_test
#SBATCH --output="slurm.%j.out"  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=90GB
##SBATCH --export=ALL  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fubo.nw@gmail.com
#SBATCH -v
#SBATCH -t 00:58:00  

module purge
module load slurm/default
module load anaconda3/5.3.0 intel/18.0/64/18.0.3.222 intel-mkl/2018.3/3/64
export PYTHONPATH=/home/bfu/work1/apps/newqutip/installation/lib/python

#export MKL_THREADING_LAYER=intel
#export OMP_NUM_THREADS=1 #1
#export MKL_NUM_THREADS=20 #28


export MKL_THREADING_LAYER=sequential
export OMP_NUM_THREADS=1 #1
export MKL_NUM_THREADS=1 #28




echo `which python`
echo `which conda`

conda env list
conda activate qutip-env3.5

export PATH=/home/bfu/.conda/envs/qutip-env3.5/bin:$PATH

echo `which python`
echo `which conda`


cd $SLURM_SUBMIT_DIR

echo Job ID is $SLURM_JOB_ID
echo Total number of cores is $SLURM_NTASKS

master=`hostname -s`
echo This job is running on host $master
echo `uname` operating system at `date`
echo ------ execution begins ------
echo "------ **************** ------"
echo ""

#python -m cProfile -s 'time' DonorAcceptorTwoModeWavepacketPrep.py 2>&1 | tee $SLURM_SUBMIT_DIR/output.log
#python DA2ModeCalcSinglePara.py 2>&1 | tee $SLURM_SUBMIT_DIR/output.log
#python -m cProfile -s 'time' test.py 2>&1 | tee $SLURM_SUBMIT_DIR/output.log
python test.py 2>&1 | tee $SLURM_SUBMIT_DIR/output.log



echo "------ **************** ------"
echo ------ execution ends ------
echo `uname` operating system at `date`

sacct  -a -j $SLURM_JOB_ID -o Job,Elapsed,AveCPU,NNodes,NCPUS,User,MaxRSS,AveRSS,State


