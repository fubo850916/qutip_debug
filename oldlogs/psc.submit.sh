#!/bin/bash  
#SBATCH -J qutiptest
#SBATCH --output="slurm.%j.out"  
##SBATCH -e slurm.%j.err
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --ntasks-per-node=28
#SBATCH --mem=120GB
##SBATCH --export=ALL  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fubo.nw@gmail.com
#SBATCH -v
#SBATCH -t 1:20:00  

module purge
module load slurm/default
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/bfu1003/project/apps/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/bfu1003/project/apps/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/bfu1003/project/apps/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/bfu1003/project/apps/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH=/home/bfu1003/project/apps/mpfr/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/bfu1003/project/apps/mpfr/include:$C_INCLUDE_PATH

export PYTHONPATH=/home/bfu1003/project/apps/bigfloat/lib/python:/home/bfu1003/project/apps/newqutip/installation/lib/python
#export PYTHONPATH=/home/bfu1003/project/apps/bigfloat/lib/python:/home/bfu1003/project/apps/qutip/lib/python

#export MKL_THREADING_LAYER=intel
export MKL_THREADING_LAYER=sequential
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo `which python`
echo `which conda`

conda env list
conda activate qutip-env3.5

cd $SLURM_SUBMIT_DIR

echo Job ID is $SLURM_JOB_ID
echo Total number of cores is $SLURM_NTASKS

master=`hostname -s`
echo This job is running on host $master
echo `uname` operating system at `date`
echo ------ execution begins ------
echo "------ **************** ------"
echo ""

python -m cProfile -s 'time' test.py 2>&1 | tee $SLURM_SUBMIT_DIR/output.log

echo "------ **************** ------"
echo ------ execution ends ------
echo `uname` operating system at `date`

sacct  -a -j $SLURM_JOB_ID -o Job,Elapsed,AveCPU,NNodes,NCPUS,User,MaxRSS,AveRSS,State


