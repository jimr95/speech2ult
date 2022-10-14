# speech2ult
Code from my dissertation "Acoustic-to-Articulatory Inversion with Extracted Tongue Contour Features from Ultrasound Tongue Imaging", submitted Aug. 2022 as part of my MSc Speech and Language Processing at The University of Edinburgh.

## Abstract
Acoustic-to-articulatory inversion (AAI) requires parallel acoustic and articulatory data, which can be difficult and expensive to acquire. Over the past decades most approaches have used electromagnetic articulography (EMA) data, which is collected by attaching sensors to a speaker’s tongue and tracking movements. However, this approach has a few drawbacks, one of which is that it can be slightly unnatural for speakers to talk with sensors on their tongue. This paper explores the use of extracted tongue contour features from ultrasound data in an AAI deep neural model that emulates the tongue tracking ability of EMA without the invasiveness. We find that with the extracted features, AAI predictions are more readable than generating raw ultrasound images, but better metrics need to be developed to compare the accuracy against the model built to generate ultrasound images.

## Acknowledgements
The primary methodology for this project (i.e. the preprocessing of data and the building of neural networks) is based on two papers that explore speaker dependent AAI with deep neural networks, one of which was trained using raw ultrasound tongue imaging data [(Porras et al., 2019)](https://arxiv.org/abs/1904.06083) and the other was trained using MRI articulator data [(Csapó, 2020)](https://arxiv.org/abs/2008.02098). The Keras implementation of the MRI experiment by Csapó is publicly available on GitHub as [speech2mri](https://github.com/BME-SmartLab/speech2mri) and has been adapted for this project after being shown to produce decent results on its intended task. This project also made use of GitHub repositories [Ultrasuite-Tools](https://github.com/UltraSuite/ultrasuite-tools) (Eshky, 2019) and [TaL-Tools](https://github.com/UltraSuite/tal-tools) (Ribeiro, 2020) for various data processing tools related to the [Tongue and Lips (TaL) corpus](https://arxiv.org/abs/2011.09804) (Ribeiro et al., 2021). Some of these tools have been added to the `tools/` directory. `tools/animate_utterance.py` has been slightly modified from its original version.

## Contents
`job_scripts/` contain various bash scripts to run preprocessing and training jobs on UoE server Eddie.
`preprocess/` contains two python scripts: 
  - `dlc_analyze.py` is used to extract DeepLabCut tongue and lip features from TAL data en masse
  -  `preprocess.py` takes a number of command line arguments and preprocesses the TAL data for neural network training and testing to be saved into pickled dictionaries.

`tools/` contains python scripts from [Ultrasuite-Tools](https://github.com/UltraSuite/ultrasuite-tools) and [TaL-Tools](https://github.com/UltraSuite/tal-tools) which are used in the preprocessing and ultrasound visualization scripts.

`train_model.py` is the main script for training, saving, and evaluating an AAI neural network. It takes the directory where the pickled preprocessed data is stored and the directory where the model should be saved to as command line arguments, along with other optional arguments.


