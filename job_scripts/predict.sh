#!/bin/sh
#
# grid engine options
#$ -N predict2
#$ -wd /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation
#$ -l h_rt=00:10:00
#$ -l h_vmem=5G
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2226889@inf.ed.ac.uk
#$ -m ea

# initialise environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate dissenv

set -euo pipefail
# set any variables
SCRATCH=/exports/eddie/scratch/s2226889
DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen
TAL=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/TaL-Corpus/TaL1/core

python make_prediction.py $SCRATCH/ffn_test2/model2/DNN_ffn_test2_2022-07-18_model.json \
    $SCRATCH/ffn_test2/data/AUD_ffn_test2_2022-07-18.pickle \
    $TAL/day2/226_aud.wav \
    --out-dir $SCRATCH/ffn_test2/model2

# model, aud_dict, aud_files