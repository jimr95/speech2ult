#!/bin/sh
#
# grid engine options
#$ -N train_dlc2ult_med
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=00:25:00
#$ -l h_vmem=35G
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
TEST_NAME=dlc2ult
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen

# make a new folder in scratch for model
mkdir -p $SCRATCH/$TEST_NAME/model_ffn_med

python $DS_HOME/dissertation/train_model.py \
    $SCRATCH/lips_test/data2 \
    $SCRATCH/$TEST_NAME/model_ffn_med \
    --dlc2ult $SCRATCH/dlc_tests/ffn_dlc_butter/data/ULT_ffn_dlc_med_2022-08-04.pickle \
    --model-type dlc2ult \
    --patience 20
# path to pickle data dir
# path to output dir
