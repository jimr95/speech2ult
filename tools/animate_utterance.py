"""
Functions to animate an ultrasound utterance.

Date: Dec 2017
Author: Aciel Eshky

Modified: Aug 2022, by Jacob Rosen

"""

import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt

from tools.read_core_files import *
from tools.ultrasound_utils import reduce_frame_rate
from tools.transform_ultrasound import transform_ultrasound


def write_images_to_disk(ult_3d, directory, title=None, aspect='auto'):
    """
    A function to write the ultrasound frames as images to a directory. The images are generated as plots without axes.
    :param ult_3d: input ultrasound object as a 3d numpy array
    :param directory: the directory to write the images to
    :param title: an optional title for the image
    :return:
    """

    plt.figure(dpi=400, figsize=(1.5, 1), facecolor='black')

    if title is not None:
        plt.title(title)

    c = ult_3d[0]
    im = plt.imshow(c.T, aspect=aspect, origin='lower', cmap='gray')
    for i in range(1, ult_3d.shape[0]):
        c = ult_3d[i]
        im.set_data(c.T)
        plt.axis("off")
        plt.savefig(directory + "/%07d.jpg" % i)


def create_video(ult_3d, frame_rate, output_video_file, out_dir, title=None, aspect='auto', save_images=False):
    """
    A function to animate an ultrasound utterance.
    :param ult_3d: input ultrasound as a 3d numpy array. Can be raw or transformed.
    :param frame_rate: which can be found in the ultrasound parameter file.
    :param output_video_file: the path/name of the output video
    :param out_dir: the directory in which to create another dir to write images to.
    :param title: an optional title for the video
    :param save_images: save images instead of deleting temp dir.
    :return:
    """
    # create temp dir for ult images within video temp dir
    if save_images:
        basename = os.path.basename(output_video_file)[:-4]
        directory = os.path.join(out_dir, f'{basename}_saved_images')
        os.mkdir(directory)
    else:
        temp_dir = tempfile.TemporaryDirectory(dir=out_dir)
        directory = temp_dir.name

    write_images_to_disk(ult_3d, directory, title=title, aspect=aspect)

    subprocess.call(
        ["ffmpeg", "-y", "-r", str(frame_rate),
         "-i", directory + "/%07d.jpg", "-vcodec", "mpeg4", "-qscale", "5", "-r",
         str(frame_rate), output_video_file])

    if not save_images:
        temp_dir.cleanup()


def crop_audio(audio_start_time, input_audio_file, output_audio_file):
    """
    A function to crop the audio.
    :param audio_start_time: taken from the ultrasound parameter file: 'TimeInSecsOfFirstFrame'
    :param input_audio_file: path/name of input audio
    :param output_audio_file: path/name of output audio
    :return:
    """
    print("cropping audio...")

    subprocess.call(
        ["ffmpeg", "-ss", str(audio_start_time), "-i", input_audio_file, output_audio_file])


def append_audio_and_video(audio_file, video_file, output_video_file):
    """
    Outputs the video file with audio.
    :param audio_file:
    :param video_file:
    :param output_video_file:
    :return:
    """
    print("appending audio to video...")

    subprocess.call(
        ["ffmpeg",
         "-i", audio_file,
         "-i", video_file,
         "-codec", "copy", "-shortest", output_video_file])


def animate_utterance(prompt_file, wave_file, ult_file, param_file, output_video_filename="out.avi", frame_rate=24,
                      background_colour=255, aspect='equal'):
    """

    :param prompt_file:
    :param wave_file:
    :param ult_file:
    :param param_file:
    :param output_video_filename:
    :param frame_rate: the video frame rate. This will be different to the ultrasound framerate
    :param background_colour: black = 0 and white = 255
    :param aspect
    :return:
    """

    # temp file names
    temp_audio_file = "cropped_audio.wav"
    temp_video_file = "video_only.avi"

    # prompt file is used for a video caption
    video_caption = parse_prompt_file(prompt_file)[0]

    # read parameter file
    param_df = parse_parameter_file(param_file=param_file)

    # use offset parameter to crop audio
    crop_audio(audio_start_time=0,
               input_audio_file=wave_file,
               output_audio_file=temp_audio_file)

    # read ultrasound, reshape it, reduce the frame rate for efficiency, and transform it
    ult = read_ultrasound_file(ult_file=ult_file)

    ult_3d = ult.reshape(-1, int(param_df['NumVectors'].value), int(param_df['PixPerVector'].value))

    x, fps = reduce_frame_rate(ult_3d=ult_3d, input_frame_rate=float(param_df['FramesPerSec'].value),
                               output_frame_rate=frame_rate)

    print("transforming raw ultrasound to world...")
    y = transform_ultrasound(x, background_colour=background_colour, num_scanlines=int(param_df['NumVectors'].value),
                             size_scanline=int(param_df['PixPerVector'].value), angle=float(param_df['Angle'].value),
                             zero_offset=int(param_df['ZeroOffset'].value), pixels_per_mm=3)

    # create video without audio
    create_video(y, fps, temp_video_file, title=video_caption, aspect=aspect)

    # append audio and video
    append_audio_and_video(temp_audio_file, temp_video_file, output_video_filename)

    # remove temporary files
    os.remove(temp_audio_file)
    os.remove(temp_video_file)

    print("Creation of video", output_video_filename, "complete.")


def animate_core_utterance(core, output_video_filename="out.avi", aspect='equal'):
    """
    A function to animate an utterance as a core object.
    :param core:
    :param output_video_filename:
    :param aspect:
    :return:
    """
    # temp file names
    temp_audio_file = "cropped_audio.wav"
    temp_video_file = "video_only.avi"

    # prompt file is used for a video caption
    video_caption = core.prompt

    # use offset parameter to crop audio
    wavfile.write(data=core.wav, rate=core.params['wav_fps'], filename=temp_audio_file)

    # create video without audio
    if core.params['ult_transformed']:
        create_video(core.ult_t, core.params['ult_fps'], temp_video_file, os.getcwd(), aspect=aspect)
    else:
        create_video(core.ult, core.params['ult_fps'], temp_video_file, os.getcwd(), aspect=aspect)
    # append audio and video
    append_audio_and_video(temp_audio_file, temp_video_file, output_video_filename)

    # remove temporary files
    os.remove(temp_audio_file)
    os.remove(temp_video_file)

    print("Creation of video", output_video_filename, "complete.")


def animate_ult(ult, param, out_path, out_dir, background_colour=0, aspect='equal', save_images=False):
    """

    :param ult_file:
    :param param_file:
    :param out_path:
    :param background_colour: black = 0 and white = 255
    :param aspect
    :return:
    """

    # transform ultrasound first
    ult_t = transform_ultrasound(ult, background_colour=background_colour, num_scanlines=param['scanlines'],
                             size_scanline=param['echos'], angle=float(param['Angle']),
                             zero_offset=int(param['ZeroOffset']), pixels_per_mm=0.5)

    # get rid of border
    ult_t = ult_t[:, 50:-50, 10:-10]

    # create video without audio
    create_video(ult_t, param['FramesPerSec'], out_path, out_dir, aspect=aspect, save_images=save_images)

    assert os.path.exists(out_path), f'ult vid {out_path} should exist'

    return out_path

def write_vid_to_disk(vid, directory, aspect='auto'):
    """
    A function to write the vid frames as images to a directory. The images are generated as plots without axes.
    :param ult_3d: input ultrasound object as a 3d numpy array
    :param directory: the directory to write the images to
    :param title: an optional title for the image
    :return:
    """

    plt.figure(dpi=300, figsize=(1.5, 1))

    c = vid[0]
    im = plt.imshow(c, aspect=aspect, origin='lower', cmap='gray')
    for i in range(1, vid.shape[0]):
        c = vid[i]
        im.set_data(c)
        plt.axis("off")
        plt.savefig(directory + "/%07d.jpg" % i, bbox_inches='tight', transparent=True, facecolor='black')

def animate_vid(vid, fps, out_path, temp_dir, aspect='equal'):
    """

    :param vid:
    :param fps: frame rate
    :param out_path:
    :param aspect
    :return:
    """
    # create temp dir to write images to
    directory = tempfile.TemporaryDirectory(dir=temp_dir)
    # write images to dir
    write_vid_to_disk(vid, directory.name, aspect=aspect)
    # create vid from images
    subprocess.call(
        ["ffmpeg", "-y", "-r", str(fps),
         "-i", directory.name + "/%07d.jpg", "-vcodec", "mpeg4", "-qscale", "4", "-r",
         str(fps), '-vf', 'transpose=0,transpose=2', out_path])
    # clear temp dir
    directory.cleanup()

    assert os.path.exists(out_path), f'lip vid {out_path} should exist'

    return out_path