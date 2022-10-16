#!/bin/sh
#
# grid engine options
#$ -N dlc2ult_eval
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=00:30:00
#$ -l h_vmem=50G
#$ -pe gpu-titanx 1
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
TEST_NAME=dlc2ult
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen

# make a new folder in scratch for model
mkdir -p $SCRATCH/$TEST_NAME/eval

python $DS_HOME/dissertation/dlc2ult_eval.py \
    $SCRATCH/ffn_dlc_ult/model/DNN_ffn_dlc_ult_2022-07-22_model.json \
    $SCRATCH/dlc2ult/model/dlc2ult_lips_test_2022-07-22_model.json \
    $SCRATCH/ffn_dlc_ult/data/AUD_ffn_dlc_ult_2022-07-22.pickle \
    $SCRATCH/ffn_dlc_ult/data/ULT_ffn_dlc_ult_2022-07-22.pickle \
    $SCRATCH/lips_test/data/ULT_lips_test_2022-07-22.pickle \
    --out-dir $SCRATCH/$TEST_NAME/eval \
    --save-images

# aai_model, dlc2ult_model, aud, dlc, ult, out-dir, save-images
