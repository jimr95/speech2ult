#!/bin/sh
#
# grid engine options
#$ -N preprocess_dlc_lstm
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=1:00:00
#$ -l h_vmem=5G
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2226889@inf.ed.ac.uk
#$ -m beas

# initialise environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate dissenv

set -euo pipefail
# set any variables
TEST_NAME=lstm_dlc
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen
TAL=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/TaL-Corpus/TaL1/core

# make a new folder in scratch to set up test
mkdir -p $SCRATCH/dlc_tests/$TEST_NAME/data

python $DS_HOME/dissertation/preprocess/preprocess.py \
    $TAL \
    $SCRATCH/dlc_tests/$TEST_NAME/data \
    --all-days\
    --file-split $SCRATCH/ffn_test2/data/FILES_ffn_test2_2022-07-18.pickle \
    --name $TEST_NAME \
    --dlc-data $SCRATCH/dlc_data \
    --dlc-ult \
    --model LSTM
# path to data dir
# path to output dir