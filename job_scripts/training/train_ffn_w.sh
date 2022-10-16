#!/bin/sh
#
# grid engine options
#$ -N train_ffn_w3
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=2:00:00
#$ -l h_vmem=30G
#$ -pe gpu-titanx 2
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2226889@inf.ed.ac.uk
#$ -m beas
#$ -P lel_hcrc_cstr_students

# initialise environment modules
. /etc/profile.d/modules.sh

module load cuda/10.2.89
module load anaconda
source activate dissenv

# need to export path to lib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

set -euo pipefail
# set any variables
TEST_NAME=ffn_w
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen

# make a new folder in scratch for model
mkdir -p $SCRATCH/new_tests/$TEST_NAME/model3

python $DS_HOME/dissertation/train_model.py \
    $SCRATCH/new_tests/$TEST_NAME/data \
    $SCRATCH/new_tests/$TEST_NAME/model3 \
    --patience 15 \
    --batch 3000
# path to pickle data dir
# path to output dir