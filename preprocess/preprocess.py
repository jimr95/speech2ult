'''
Preprocess TAL data for AAI model, splitting it into training, validation, and test sets.
The input (audio) and output (ultrasound/DLC features) are saved as two different dictionaries.
There was on option to included lips in preprocessing that isn't fully developed yet.

Written by Jacob Rosen
Aug 2022

Loosely based on preprocessing implementation of Csapó T.G., ,,Speaker dependent acoustic-to-articulatory inversion using real-time MRI of the vocal tract'', accepted at Interspeech 2020
'''

import glob
import logging
import numpy as np
import os
import argparse
import pickle
import datetime
import random
import librosa
import pandas as pd
import scipy.signal as signal

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.transform import resize
# this is not the correct path for here but on eddie pwd is dissertation
import sys
# TODO add correct path to tools
sys.path.insert(0, '/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2226889_Jacob_Rosen/dissertation/tools')
# sys.path.insert(0, '/Users/jacobrosen/Desktop/Edinburgh/Dissertation/project_code/speech2ult/tools')
from tal_io import read_ultrasound_tuple, read_ultrasound_param
from voice_activity_detection import detect_voice_activity, separate_silence_and_speech
import utils

def get_args():

    parser = argparse.ArgumentParser(description='Data preprocessing')

    # data directories
    parser.add_argument('data', help='path to data directory')
    parser.add_argument('output_dir', help='where to output files')
    parser.add_argument('--file-split', help='path to a pickled database file split to load')
    parser.add_argument('--all-days', action='store_true', help='go through day directories')
    parser.add_argument('--test', action='store_true', help='test on only first ten files')
    parser.add_argument('--dlc-data', default=None, help='path to DLC data')

    # ultrasound params
    parser.add_argument('--ult-fps', type=int, default=60, help='expected fps for ultrasound')
    parser.add_argument('--ult-size', type=int, nargs=2, default=[64, 128], help='two ints rep size of ult scans-echos')
    parser.add_argument('--dlc-ult', action='store_true', help='use dlc for ult')
    parser.add_argument('--butter', action='store_true', help='apply butterworth filter to dlc')
    parser.add_argument('--median', action='store_true', help='apply median filter to dlc')

    # audio params
    parser.add_argument('--aud-sr', type=int, default=20000, help='audio sampling rate')
    parser.add_argument('--n-feat', type=int, default=20, help='number fo features to use')
    parser.add_argument('--deltas', action='store_true', help='use deltas or not')

    # lip params
    parser.add_argument('--lips-in', action='store_true', help='process lips as input')
    parser.add_argument('--lips-out', action='store_true', help='process lips as output')

    # the type of model data will be used for
    parser.add_argument('--model', default='DNN', help='type of model data formated for')
    parser.add_argument('--lookback', type=int, default=10, help='lookback for LSTM')
    parser.add_argument('--name', default=None, help='name of output files')
    parser.add_argument('--window', action='store_true', help='give ffn a window of audio features')

    args = parser.parse_args()
    return args

def parse_files(args):
    """
    Get a list of all file paths corresponding to aud.ult then split files in training, valid, and test sets.
    Based on how TAL data is stored.
    Returns a dictionary of lists.
    """
    # TaL1 has 1246 read utterance files - 206 from day1 dont have video sync so in total 1039
    # can use 80:20 train-test split
    # 1039 * 0.2 = 208
    # valid: 104
    # test: 104
    logging.info(f'Parsing files from {args.data}')
    files_dict = dict()
    files_dict['all'] = []
    dir_list = []
    if args.all_days:
        dir_list = [os.path.join(args.data, day) for day in ['day2', 'day3', 'day4', 'day5', 'day6']]
    else:
        dir_list.append(args.data)

    # this gives the entire path to all ult files
    files_dict['all'] = [file for dir in dir_list for file in glob.glob(os.path.join(dir, '*aud.ult'))]
    if args.test:
        files_dict['all'] = files_dict['all'][0:10]

    # randomize file order
    random.seed(17)
    random.shuffle(files_dict)

    # count split for 80-10-10
    total_files = len(files_dict['all'])
    split = total_files // 10
    files_dict['valid'] = files_dict['all'][0:split]
    files_dict['test'] = files_dict['all'][split:split*2]
    files_dict['train'] = files_dict['all'][split*2:]
    logging.info(f'Total files used in database: {total_files}')
    logging.info(f'Train-Valid-Test split: {total_files - split*2}-{split}-{split}')

    # pick one file from test to make predictions on for all models and compare
    files_dict['test_file'] = {'name':files_dict['test'][0]}
    logging.info(f'Prediction test file: {files_dict["test"][0]}')

    return files_dict

def resize_ult(ult, n_scan, n_echo):
    """
    resize the ult to new size n_scan x n_echo. adapted from ultrasuite tools core.py
    :return: ult
    """

    resized = []
    for i, image in enumerate(ult):
        temp = resize(image, output_shape=(n_scan, n_echo), order=0, mode='reflect', clip=False,
                      preserve_range=True, anti_aliasing=False)
        temp = temp.round().astype(int)
        resized.append(temp)

    ult = np.array(resized)
    return ult

def load_ult(file, params):
    """load, downsample and resize ultrasound, returned with the sync offset in seconds"""
    ult, param_file = read_ultrasound_tuple(file, shape='3d', cast=None, truncate=None)
    offset = param_file['TimeInSecsOfFirstFrame']
    # current n_frames * target_fps/current_fps
    target_frames = int(ult.shape[0] * params['ult_fps'] / param_file['FramesPerSec'])
    ult = utils.resize(ult, target_frames)
    # resize dimensions
    ult = resize_ult(ult, params['n_scan'], params['n_echo'])


    return ult, offset

def clean_data(data):
    """Clean the dlc pandas dataset and return np array of features"""
    body_parts = data.loc['bodyparts'].values + '_' + data.loc['coords'].values
    data.columns = body_parts
    data.drop(index=['bodyparts', 'coords'], inplace=True)
    del_col = [col for col in data.columns if col.endswith('likelihood')]
    data.drop(columns=del_col, inplace=True)
    # get rid of columns for hyoid, mandible and short tendon as well
    if 'hyoid_x' in data.columns:
        del_parts = ['hyoid_x', 'hyoid_y','mandible_x', 'mandible_y', 'shortTendon_x', 'shortTendon_y']
        data.drop(columns=del_parts, inplace=True)

    return data.values

def apply_butter(dlc_feats):
    """apply butterworth filter to each column (body part)"""
    filtered = np.empty_like(dlc_feats)
    fc = 10  # Cut-off frequency of the filter
    w = fc / (60 / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')

    for col in range(dlc_feats.shape[1]):
        filtered[:, col] = signal.filtfilt(b, a, dlc_feats[:, col])

    return filtered

def apply_medfilt(dlc_feats):
    """apply median filter to each column (body part)"""
    filtered = np.empty_like(dlc_feats)
    for col in range(dlc_feats.shape[1]):
        filtered[:, col] = signal.medfilt(dlc_feats[:, col], 7)

    return filtered

def load_dlc_ult(file_path, args):
    """
    Needs to read in the dlc features corresponding to specified file_path from the args directory and return the
    features and the offset from the ult params. This is specidifc to how dlc files are stored.
    """
    params = read_ultrasound_param(file_path + '.param')
    offset = params['TimeInSecsOfFirstFrame']
    # no need to downsample or resize or sync since already done before DLC
    path, basename = os.path.split(file_path)
    day = os.path.basename(path)
    csv_file = glob.glob(os.path.join(args.dlc_data, day, f'{day}_{basename}_ult*.csv'))[0]
    # csv_file = glob.glob(os.path.join(args.dlc_data, 'day2', f'day2_{basename}_ult*.csv'))[0]
    data = pd.read_csv(csv_file, index_col=0)
    ult_features = clean_data(data).astype(float)

    # apply smoothing filters to dlc
    if args.butter:
        ult_features = apply_butter(ult_features)
    if args.median:
        ult_features = apply_medfilt(ult_features)

    return ult_features, offset

def load_dlc_lips(file_path, args):
    """
    Needs to read in the dlc features corresponding to specified file_path from the args directory and return the
    features for lips. This is specidifc to how dlc files are stored.
    """
    # no need to downsample or resize or sync since already done before DLC
    path, basename = os.path.split(file_path)
    day = os.path.basename(path)
    csv_file = glob.glob(os.path.join(args.dlc_data, day, f'{day}_{basename}_lips*.csv'))[0]
    # csv_file = glob.glob(os.path.join(args.dlc_data, 'day2', f'day2_{basename}_lips*.csv'))[0]
    data = pd.read_csv(csv_file, index_col=0)
    lip_feats = clean_data(data)

    return lip_feats

def get_aud_feats(wav, params):
    """get the audio features from a loaded wav, return as (frames, features)"""
    n_mfcc = params['n_feat']//2 if params["deltas"] else params['n_feat']
    mfccs = librosa.feature.mfcc(
        y=wav,
        sr=params['sr'],
        n_mfcc=n_mfcc,
        hop_length=params['win_shift'],
        n_fft=params['win_len']
    )
    if params['deltas']:
        deltas = librosa.feature.delta(mfccs)
        feats = np.concatenate((mfccs,deltas))
        return feats.T
    else:
        return mfccs.T

def save_files(args, file_dict, ult_data, aud_data, lip_data, name):
    """Save the file_dict and other data requested along with a file with program params"""
    logging.info(f'Saving data in {args.output_dir}')
    # save file_dict to know train-test split
    files_path = os.path.join(args.output_dir, f'FILES_{name}.pickle')
    with open(files_path, "wb") as file:
        pickle.dump(file_dict, file, pickle.HIGHEST_PROTOCOL)
    # save ult
    ult_p_path = os.path.join(args.output_dir, f'ULT_{name}.pickle')
    with open(ult_p_path, "wb") as file:
        pickle.dump(ult_data, file, pickle.HIGHEST_PROTOCOL)
    # save aud
    aud_p_path = os.path.join(args.output_dir, f'AUD_{name}.pickle')
    with open(aud_p_path, "wb") as file:
        pickle.dump(aud_data, file, pickle.HIGHEST_PROTOCOL)

    # save param dicts seperately as well
    param_path = os.path.join(args.output_dir, f'PARAM_DICTS_{name}.pickle')
    with open(param_path, "wb") as file:
        pickle.dump((ult_data[1], aud_data[1]), file, pickle.HIGHEST_PROTOCOL)

    # save lips
    if args.lips_in or args.lips_out:
        lip_p_path = os.path.join(args.output_dir, f'LIP_{name}.pickle')
        with open(lip_p_path, "wb") as file:
            pickle.dump(lip_data, file, pickle.HIGHEST_PROTOCOL)

    # log params
    log_path = os.path.join(args.output_dir, f'INFO_{name}.txt')
    with open(log_path, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"Argument {arg}: {value}\n")
        # also keep track of which files are being used for tests
        file.write(f'***Test files***\n')
        for name in file_dict['test']:
            file.write(name + '\n')

    logging.info('Save complete')

def normalize(dict, type):
    """Takes a dict of features and returns normalized data using the type of normalization specified"""
    scaler = MinMaxScaler(feature_range=(0,1)) if type == 'minmax' else StandardScaler(with_mean=True, with_std=True)
    dict['train'] = scaler.fit_transform(dict['train'])
    dict['valid'] = scaler.transform(dict['valid'])
    dict['test'] = scaler.transform(dict['test'])

    return dict, scaler

def reshape_for_LSTM(aud_in, lookback):
    """
    Modified version of Csapo's implimentation. Main idea is that each ultrasound frame should match up with
    lookback number of previous audio frames.
    """
    # get dims of aud_in
    frames, n_feats = aud_in.shape

    # create new aud array with space for lookback
    aud_out = np.empty((frames - (lookback - 1), lookback, n_feats))

    # populate new arrays
    for i in range(frames - lookback): # dont need -1 here becasue index starts at 0
        aud_out[i] = aud_in[i:(i+lookback)]

    return aud_out

def reshape_audio_dict(aud_dict, aud_params, ult_dict):
    """takes audio dict and reshapes each array for lstm, checking it matches size fo ult_dict"""
    new_dict = dict()
    for train_valid in ['train', 'valid', 'test']:
        new_dict[train_valid] = np.empty((aud_dict[train_valid].shape[0], aud_params['lb'], aud_params['n_feat']))
        n_frames = 0
        for i in range(len(aud_params['files'][train_valid])-1):
            start = aud_params['files'][train_valid][i]
            end = aud_params['files'][train_valid][i+1]
            aud_in = aud_dict[train_valid][start:end]
            new_aud = reshape_for_LSTM(aud_in, aud_params['lb'])
            new_dict[train_valid][n_frames:n_frames + new_aud.shape[0]] = new_aud
            n_frames += new_aud.shape[0]
        new_dict[train_valid] = new_dict[train_valid][:n_frames]
        assert new_dict[train_valid].shape[0] == ult_dict[train_valid].shape[0], 'the dictionaries should have n_frames'

    return new_dict

def reshape_lip_dict(lip_dict, lip_params, aud_params, aud_dict):
    """takes lip dict and reshapes each array for lstm, checking it matches size with aud_dict"""
    new_dict = dict()
    for train_valid in ['train', 'valid', 'test']:
        new_dict[train_valid] = np.empty((lip_dict[train_valid].shape[0], lip_params['lb'], lip_params['n_feat']))
        n_frames = 0
        for i in range(len(aud_params['files'][train_valid])-1):
            start = aud_params['files'][train_valid][i]
            end = aud_params['files'][train_valid][i+1]
            lips_in = lip_dict[train_valid][start:end]
            new_lips = reshape_for_LSTM(lips_in, aud_params['lb'])
            new_dict[train_valid][n_frames:n_frames + new_lips.shape[0]] = new_lips
            n_frames += new_lips.shape[0]
        new_dict[train_valid] = new_dict[train_valid][:n_frames]
        assert new_dict[train_valid].shape[0] == aud_dict[train_valid].shape[0], 'the dicts should have same n_frames'


    return new_dict

def apply_sync(ult, ult_fps, wav, wav_sr, lips=None):
    """all should be aligned at begining, need to trim the ends and return"""

    ult_dur = ult.shape[0] / ult_fps
    wav_dur = wav.shape[0] / wav_sr
    lips_dur = float('inf')
    if lips is not None:
        lips_dur = lips.shape[0] / ult_fps

    min_sec = np.min((ult_dur, wav_dur, lips_dur))
    ult_end = int(round(ult_fps * min_sec))
    wav_end = int(round(wav_sr * min_sec))

    if lips is not None:
        return ult[:ult_end], wav[:wav_end], lips[:ult_end]
    else:
        return ult[:ult_end], wav[:wav_end]

def apply_vad(ult, ult_fps, wav, wav_sr, lips=None):
    """Apply voice activity detection - ie endpointing to remove silence. From Aciel Eshky ultrasuite tools"""
    # get time segments
    time_segments = detect_voice_activity(wav,wav_sr)

    # apply to wav
    silence, speech = separate_silence_and_speech(wav, wav_sr, time_segments)
    wav = speech

    # apply to ult
    silence, speech = separate_silence_and_speech(ult, ult_fps, time_segments)
    ult = speech

    if lips is not None:
        silence, speech = separate_silence_and_speech(lips, ult_fps, time_segments)
        lips = speech
        return ult, wav, lips
    else:
        return ult, wav

def reshape_for_window(aud_in):
    """reshape audio from (frames, features) to (frames, 5, features)"""
    # get dims of aud_in
    frames, n_feats = aud_in.shape

    # create new aud array with space for window
    aud_out = np.empty((frames - 4, 5, n_feats))

    # populate new arrays
    for i in range(0, frames-5):  # i the first frame in the window
        aud_out[i] = aud_in[i:i+5]

    return aud_out

def get_window(aud_dict, aud_params, ult_dict):
    """takes audio dict and reshapes each array for ffn with 5 frame window, checking it matches size of ult_dict"""
    new_dict = dict()
    for train_valid in ['train', 'valid', 'test']:
        new_dict[train_valid] = np.empty((aud_dict[train_valid].shape[0], 5, aud_params['n_feat']))
        n_frames = 0
        for i in range(len(aud_params['files'][train_valid]) - 1):
            start = aud_params['files'][train_valid][i]
            end = aud_params['files'][train_valid][i + 1]
            aud_in = aud_dict[train_valid][start:end]
            new_aud = reshape_for_window(aud_in)
            new_dict[train_valid][n_frames:n_frames + new_aud.shape[0]] = new_aud
            n_frames += new_aud.shape[0]
        new_dict[train_valid] = new_dict[train_valid][:n_frames]
        assert new_dict[train_valid].shape[0] == ult_dict[train_valid].shape[0], 'the dictionaries should have n_frames'

        # flatten the window
        new_dict[train_valid] = new_dict[train_valid].reshape(-1, aud_params['n_feat'] * 5)

    return new_dict

def load_files(file_dict, args):
    """Load each respective file into its dictionary and return with params"""
    # initialize all dicts
    ult_dict = dict()
    ult_params = {
        "ult_fps": args.ult_fps,
        "n_scan": args.ult_size[0],
        "n_echo": args.ult_size[1],
        "n_feat": 22 if args.dlc_ult else args.ult_size[0] * args.ult_size[1]
    }
    aud_dict = dict()
    aud_params = {
        "sr": args.aud_sr,
        "win_len": int(args.aud_sr * 0.02), # 20ms <- sr * 0.02
        "win_shift": int(args.aud_sr * (1/ult_params["ult_fps"])), # sr * (1/ult_fps) to get samples per ult frame
        "n_feat": args.n_feat*2 if args.deltas else args.n_feat,
        "deltas": args.deltas,
        "lb": args.lookback if args.model == 'LSTM' else None,
        "files": dict(),
        'window': 5 if args.window else None


    }
    lip_dict = dict()
    lip_params = {
        "n_feat": 16, # 8 positions, with x-y coordinates
        "lb": args.lookback if args.model == 'LSTM' else None
    }

    logging.info('Loading files into dictionaries')

    for train_valid in ['train', 'valid', 'test']:
        n_files = len(file_dict[train_valid])
        n_max_ult_frames = n_files * 700 # most are around 500 frames
        aud_frames = 0
        ult_frames = 0

        # establish array size for ult data
        if args.dlc_ult:
            ult_dict[train_valid] = np.empty((n_max_ult_frames, 22)) # 11 points times x,y coords
        else:
            ult_dict[train_valid] = np.empty((n_max_ult_frames, ult_params['n_scan'], ult_params['n_echo']))

        # establish array size for audio data
        aud_dict[train_valid] = np.empty((n_max_ult_frames, aud_params['n_feat']))
        # need to keep track of file lengths to reshape for LSTM
        aud_params['files'][train_valid] = [0]

        # establish array size for lip data
        if args.lips_in:
            lip_dict[train_valid] = np.empty((n_max_ult_frames, 16))

        # load in each file one-by-one
        for file in file_dict[train_valid]:
            if args.dlc_ult:
                ult, offset = load_dlc_ult(file[:-4], args)
            else:
                ult, offset = load_ult(file[:-4], ult_params)
            wav, sr = librosa.load(file[:-4] + '.wav', offset=offset, sr=aud_params['sr'])
            if args.lips_in:
                lip_feats = load_dlc_lips(file[:-4], args)

            # sync signals (wav already offset, just need to trim end) then VAD ie endpointing
            if args.lips_in:
                ult, wav, lip_feats = apply_sync(ult, ult_params['ult_fps'], wav, sr, lips=lip_feats)
                ult, wav, lip_feats = apply_vad(ult, ult_params['ult_fps'], wav, sr, lips=lip_feats)
            else:
                ult, wav = apply_sync(ult, ult_params['ult_fps'], wav, sr)
                ult, wav = apply_vad(ult, ult_params['ult_fps'], wav, sr)

            # get aud_features
            aud_features = get_aud_feats(wav, aud_params) # shape: (frames, n_feat)


            # make sure they have same number of frames
            file_n_frames = np.min((aud_features.shape[0], ult.shape[0]))
            if args.lips_in:
                file_n_frames = np.min((file_n_frames, lip_feats.shape[0]))
                lip_feats = lip_feats[:file_n_frames]
            ult = ult[:file_n_frames]
            aud_features = aud_features[:file_n_frames]


            # reshape now for LSTM ultrasound, otherwise will need to redo the whole thing later
            if args.model == 'LSTM':
                ult = ult[(args.lookback - 1):]
            elif args.window: # going to use window size of 5 so remove 2 frames from each side
                ult = ult[2:-2]

            # once data from file is loaded and processed add to dicts
            ult_dict[train_valid][ult_frames:ult_frames + ult.shape[0]] = ult
            aud_dict[train_valid][aud_frames:aud_frames + file_n_frames] = aud_features
            if args.lips_in:
                lip_dict[train_valid][aud_frames:aud_frames + file_n_frames] = lip_feats

            aud_frames += file_n_frames
            ult_frames += ult.shape[0]
            # add file_n_frames to list for LSTM backtracking
            aud_params['files'][train_valid].append(aud_frames)

        # log info about each dictionary
        logging.info(f'Number of frames for {train_valid}: {ult_frames}')

        # get rid of empty parts of array and reshape ultrasound to be 2d (frames, scanlines * echos)
        ult_dict[train_valid] = ult_dict[train_valid][0:ult_frames]
        aud_dict[train_valid] = aud_dict[train_valid][0:aud_frames]
        if args.lips_in:
            lip_dict[train_valid] = lip_dict[train_valid][0:aud_frames]

        if not args.dlc_ult:
            ult_n = ult_params['n_scan'] * ult_params['n_echo']
            ult_dict[train_valid] = ult_dict[train_valid].reshape(-1, ult_n)

    # normalize and save scalars in param dicts
    ult_dict, ult_params['scalar'] = normalize(ult_dict, 'minmax') # can minmax dlc also and use inverse for prediction
    aud_dict, aud_params['scalar'] = normalize(aud_dict, 'standard')
    if args.lips_in:
        lip_dict, lip_params['scalar'] = normalize(lip_dict, 'standard')

    # reshape audio for LSTM
    if args.model == 'LSTM':
        aud_dict = reshape_audio_dict(aud_dict, aud_params, ult_dict)
        if args.lips_in:
            lip_dict = reshape_lip_dict(lip_dict, lip_params, aud_params, aud_dict)
    elif args.window:
        aud_dict = get_window(aud_dict, aud_params, ult_dict)
        logging.info(f'Using audio frame window of 5')
        aud_params['n_feat'] = aud_params['n_feat'] * 5

    logging.info('Feature extraction and normalization complete')
    logging.info(f'LSTM: {args.model == "LSTM"}, with lookback: {args.lookback}')
    logging.info(f'Shape of aud train array {aud_dict["train"].shape}')

    return [(ult_dict, ult_params), (aud_dict, aud_params), (lip_dict, lip_params)]

def main(args):
    # collect dictionary of all files, split into training, dev, testing
    # load a previous file_dict
    # file_dict has lists of files train, test, dev, valid, test_file
    if args.file_split:
        with open(args.file_split, 'rb') as file:
            file_dict = pickle.load(file)
        logging.info(f'Loaded in file_dict {args.file_split}')
    else:
        file_dict = parse_files(args)

    # load files and params into dictionaries - these vars are tuples with dict and params
    ult_data, aud_data, lip_data = load_files(file_dict, args)

    # come up with name for files
    name = f'{args.name if args.name else args.model}' + '_{date:%Y-%m-%d}'.format(date=datetime.datetime.now())

    # pickel dictionaries
    save_files(args, file_dict, ult_data, aud_data, lip_data, name)

if __name__ == "__main__":
    args = get_args()

    # check both directories are valid
    if os.path.exists(args.data) and os.path.exists(args.output_dir):
        pass
    else:
        raise Exception('one of the directories provided does not exist')

    log_path = os.path.join(args.output_dir, 'preprocess.log')
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    main(args)
