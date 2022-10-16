#!/bin/sh
#
# grid engine options
#$ -N train_ffn2_model5
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=02:00:00
#$ -l h_vmem=35G
#$ -pe gpu-titanx 2
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2226889@inf.ed.ac.uk
#$ -m beas

# initialise environment modules
. /etc/profile.d/modules.sh

module load cuda/10.2.89
module load anaconda
source activate dissenv

# need to export path to lib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

set -euo pipefail
# set any variables
TEST_NAME=ffn_test2
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen

# make a new folder in scratch for model
mkdir -p $SCRATCH/$TEST_NAME/model5

python $DS_HOME/dissertation/train_model.py \
    $SCRATCH/$TEST_NAME/data \
    $SCRATCH/$TEST_NAME/model5 \
    --profile
# path to pickle data dir
# path to output dir
