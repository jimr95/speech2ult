'''
Extracts tongue and lip contour features from each video in TAL dataset.

Written by Jacob Rosen
Aug 2022

Feature extraction uses the DeepLabCut models built by Wrench, A. and Balch-Tomes, J. (2022) (https://www.mdpi.com/1424-8220/22/3/1133) (https://doi.org/10.3390/s22031133)
https://github.com/articulateinstruments/DeepLabCut-for-Speech-Production
'''

import glob
import os
import argparse
import sys
import tempfile
import deeplabcut
import tensorflow as tf

# TODO add correct path to tools
sys.path.insert(0, '/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation/tools')
# sys.path.insert(0, '/Users/jacobrosen/Desktop/Edinburgh/Dissertation/project_code/speech2ult/tools')
import utils
from tal_io import read_ultrasound_tuple, read_waveform, read_video
from animate_utterance import animate_ult, animate_vid

def get_args():
    parser = argparse.ArgumentParser(description='Create temp video files, then extract DLC features')

    # data directories
    parser.add_argument('data', help='path to data directory')
    parser.add_argument('out_dir', help='path to output directory')
    parser.add_argument('ult_config', help='path to ult config file')
    parser.add_argument('lip_config', help='path to lip config file')
    parser.add_argument('--all-days', action='store_true', help='go through day directories')
    parser.add_argument('--test', action='store_true', help='run test on a single file')

    args = parser.parse_args()
    return args

def analyze_vids(file, name, args):
    """Create temporary ult and lip vids for each file and analyze them with deeplabcut"""

    # create temp dir
    temp_dir = tempfile.TemporaryDirectory(dir=args.out_dir)


    # create vids
    ult_path, lip_path = create_vids(file, name, temp_dir.name)
    # ult_path, lip_path = create_vids(file, name, args.out_dir)
    # check vids are actually created
    # if os.path.exists(ult_path) and os.path.exists(lip_path):
    #     print('vids actually exist!')

    # check gpu
    n_gpu = len(tf.config.list_physical_devices('GPU'))
    n_gpu = n_gpu if n_gpu > 0 else None

    # analyze
    deeplabcut.analyze_videos(args.lip_config, [lip_path], shuffle=1, gputouse=n_gpu, save_as_csv=True,
                              destfolder=args.out_dir)
    deeplabcut.analyze_videos(args.ult_config, [ult_path], shuffle=1, gputouse=n_gpu, save_as_csv=True,
                              destfolder=args.out_dir)

    # # test if the analysis is good
    # deeplabcut.create_labeled_video(args.lip_config, [lip_path], shuffle=1, draw_skeleton=True)
    # deeplabcut.create_labeled_video(args.ult_config, [ult_path], shuffle=1, draw_skeleton=True)

    # delete temp dir
    temp_dir.cleanup()

def create_vids(file, name, dir):
    """Create temporary ult and lip videos for each file, returning the path names for each file"""
    # read input data
    ult, params = read_ultrasound_tuple(file[:-4], shape='3d', cast=None, truncate=None)
    wav, wav_sr = read_waveform(file[:-4] + '.wav')
    vid, meta = read_video(file[:-4], shape='3d', cast=None)

    # fps
    ult_fps = params['FramesPerSec']
    vid_fps = meta['fps']

    # trim streams to parallel, and then downsample them
    # then we animate the ultrasound, and re-create the lip video
    ult, vid, wav = utils.trim_to_parallel_streams(ult, vid, wav, params, meta, wav_sr)
    ult, vid = utils.downsample(ult, vid, ult_fps, vid_fps, target_fps=60)
    params['FramesPerSec'] = 60
    meta['fps'] = 60

    # animate
    ult_path = animate_ult(ult, params, os.path.join(dir, name + '_ult.avi'), dir)
    lip_path = animate_vid(vid, 60, os.path.join(dir, name + '_lips.avi'), dir)

    return ult_path, lip_path

def main(args):

    # collect files to process
    dir_list = []
    if args.all_days:
        dir_list = [os.path.join(args.data, day) for day in ['day2', 'day3', 'day4', 'day5', 'day6']]
    else:
        dir_list.append(args.data)
    all_files = [file for dir in dir_list for file in glob.glob(os.path.join(dir, '*aud.ult'))]

    for file in all_files:
        # get name
        basename = os.path.basename(file)[:-4]
        day = os.path.basename(os.path.dirname(file))
        out_name = day + '_' + basename

        # analyze videos
        analyze_vids(file, out_name, args)

        # test on only one file
        if args.test:
            break

if __name__ == "__main__":
    args = get_args()
    main(args)
