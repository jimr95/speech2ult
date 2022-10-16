#!/bin/sh
#
# grid engine options
#$ -N dlc_analyze
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=24:00:00
#$ -l h_vmem=20G
#$ -pe gpu-titanx 1
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2226889@inf.ed.ac.uk
#$ -m beas

# initialise environment modules
. /etc/profile.d/modules.sh

module load cuda/10.2.89
module load anaconda
source activate slptorch

set -euo pipefail
# set any variables
TEST_NAME=dlc_data
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen
TAL=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/TaL-Corpus/TaL1/core

# make a new folder in scratch to set up test
mkdir -p $SCRATCH/$TEST_NAME

python $DS_HOME/dissertation/preprocess/dlc_analyze.py \
    $TAL \
    $SCRATCH/$TEST_NAME \
    $DS_HOME/dissertation/DeepLabCut-for-Speech-Production/Ultrasound/config.yaml \
    $DS_HOME/dissertation/DeepLabCut-for-Speech-Production/Lips/config.yaml \
    --all-days
# path to data dir
# path to output dir
# ult config
# lip config

