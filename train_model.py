'''
Train and evaluate AAI models using preprocessed TAL data.

Written by Jacob Rosen
Aug 2022

Based on Keras implementation of Csapó T.G., ,,Speaker dependent acoustic-to-articulatory inversion using real-time MRI of the vocal tract'', accepted at Interspeech 2020
'''

import argparse
import logging
import pickle
import os
import datetime
import glob
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

# Use avaible GPU
# TODO what is all this?
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
# from tf.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
config.log_device_placement=True
set_session(tf.compat.v1.Session(config=config))

# # Disable V2 behavior
# tf.compat.v1.disable_v2_behavior()

def get_args():
    parser = argparse.ArgumentParser(description='Train FC-DNN')

    # data directories
    parser.add_argument('data', help='directory containing pickle files')
    parser.add_argument('model_dir', help='path to dir storing model')
    parser.add_argument('--lips-in', action='store_true', help='use lips data in data dir')
    parser.add_argument('--dlc2ult', default=None, help='path to dlc ult pickle to build model')

    # model name
    parser.add_argument('--model-type', default='DNN', help='type of model')

    # training params
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch', type=int, default=128, help='training batch size')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stop')
    parser.add_argument('--lr', type=float, default= 1e-3, help='initial learining rate for optimizer')
    parser.add_argument('--no-stop', action='store_false', help='take out early stopping')
    parser.add_argument('--profile', action='store_true', help='log tensorboard for profiling')


    args = parser.parse_args()
    return args

def SSIMLoss(y_true, y_pred):
    """https://stackoverflow.com/questions/57357146/use-ssim-loss-function-with-keras"""
    y_true = tf.reshape(y_true, (-1, 64, 128, 1))
    y_pred = tf.reshape(y_pred, (-1, 64, 128, 1))

    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

class ActivationHistogramCallback(tf.keras.callbacks.Callback):
    """Output activation histograms."""

    def __init__(self, layers, log_dir):
        """Initialize layer data."""
        super().__init__()

        self.layers = layers
        self.batch_layer_outputs = {}
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = tf.Variable(0, dtype=tf.int64)

    def set_model(self, _model):
        """Wrap layer calls to access layer activations."""
        for layer in self.layers:
            self.batch_layer_outputs[layer] = tf_nan(layer.output.dtype)

            def outer_call(inputs, layer=layer, layer_call=layer.call):
                outputs = layer_call(inputs)
                self.batch_layer_outputs[layer].assign(outputs)
                return outputs

            layer.call = outer_call

    def on_train_batch_end(self, _batch, _logs=None):
        """Write training batch histograms."""
        with self.writer.as_default():
            for layer, outputs in self.batch_layer_outputs.items():
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                tf.summary.histogram(f"{layer.name}/activation", outputs, step=self.step)

        self.step.assign_add(1)

def tf_nan(dtype):
    """Create NaN variable of proper dtype and variable shape for assign()."""
    return tf.Variable(float("nan"), dtype=dtype, shape=tf.TensorShape(None))

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def _log_gradients(self, epoch):
        step = tf.cast(epoch, dtype=tf.int64)
        writer = self._train_writer
        # writer = self._get_writer(self._train_run_name)

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data to calculate the gradients
            _x_batch = self.validation_data[0][:100]
            _y_batch = self.validation_data[1][:100]

            g.watch(tf.convert_to_tensor(_x_batch))
            _y_pred = self.model(_x_batch)  # forward-propagation
            loss = self.model.loss(y_true=_y_batch, y_pred=_y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=step)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)

def build_DNN_model(ult_params, aud_params):
    """Build the FC-DNN taken from Csapo. 5 hidden layers, each with 1000 units"""

    logging.info('Building FCDNN model')
    model = Sequential()
    model.add(Dense(1000, input_shape=(aud_params['n_feat'],), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(ult_params['n_feat'], kernel_initializer='normal', activation='linear'))

    return model

def build_LSTM_model(ult_params, aud_params):
    """Build LSTM taken from Csapo. 3 fully connected layers, followed by 2 LSTM layers with 575 units each."""

    logging.info('Building LSTM model')
    model = Sequential()
    model.add(TimeDistributed(Dense(575, kernel_initializer='normal', activation='relu', input_shape=(aud_params['lb'],
        aud_params['n_feat']))))
    model.add(TimeDistributed(Dense(575, kernel_initializer='normal', activation='relu')))
    model.add(TimeDistributed(Dense(575, kernel_initializer='normal', activation='relu')))

    model.add(LSTM(575, kernel_initializer='normal', activation='relu', return_sequences=True))
    model.add(LSTM(575, kernel_initializer='normal', activation='relu', return_sequences=False))

    model.add(Dense(ult_params['n_feat'], kernel_initializer='normal', activation='linear'))

    return model

def train_model(model, model_path, ult_dict, aud_dict, epochs, batch, patience, early_stop=True, profile=False):

    logging.info(f'Batch size: {batch}. Ealy stop: {early_stop}. Patience: {patience}.')
    logging.info(f'Starting training at: {"{date:%H:%M:%S}".format(date=datetime.datetime.now())}')

    # early stopping to avoid over-training
    # csv logging of loss
    # save best model

    callbacks = [CSVLogger(model_path + '.csv', append=True, separator=';'),
                 ModelCheckpoint(model_path + '_weights.h5', monitor='val_loss')]
    if early_stop:
        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        callbacks.append(early)

    if profile:
        # tensorboard for tracking loss and weights
        dir = os.path.dirname(model_path)
        log_dir = os.path.join(dir, f'log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        ten = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # ten = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads =True)
        # act = ActivationHistogramCallback(model.layers, log_dir)
        # grad = ExtendedTensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(ten)

    # run training
    history = model.fit(aud_dict['train'], ult_dict['train'],
                        epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
                        validation_data=(aud_dict['valid'], ult_dict['valid']),
                        callbacks=callbacks)

    logging.info(f'Training complete at: {"{date:%H:%M:%S}".format(date=datetime.datetime.now())}')
    logging.info(f'Log information in {model_path + ".csv"}')

def save_model(model, model_path):
    model_json = model.to_json()
    with open(model_path + '_model.json', "w") as json_file:
        json_file.write(model_json)
    logging.info(f'Model saved to {model_path + "_model.json"}')

def main(args):

    # load in the data, assumes only one of each file in dir
    ult_path = glob.glob(os.path.join(args.data, 'ULT*.pickle'))[0]
    aud_path = glob.glob(os.path.join(args.data, 'AUD*.pickle'))[0]
    if args.dlc2ult:
        aud_path = args.dlc2ult

    with open(ult_path, 'rb') as file:
        ult_dict, ult_params = pickle.load(file)
    with open(aud_path, 'rb') as file:
        aud_dict, aud_params = pickle.load(file)
    logging.info(f'Data loaded in from:\n{ult_path}\n{aud_path}')

    # make sure n_frames align
    if args.dlc2ult:
        for i in ['train', 'valid', 'test']:
            assert ult_dict[i].shape[0] == aud_dict[i].shape[0], 'the dicts need to have same n frames'

    if args.lips_in:
        lip_path = glob.glob(os.path.join(args.data, 'LIP*.pickle'))[0]
        with open(lip_path, 'rb') as file:
            lip_dict, lip_params = pickle.load(file)
        logging.info(f'Lip data loaded in.')
        for train_valid in ['train', 'valid', 'test']:
            aud_dict[train_valid] = np.concatenate((aud_dict[train_valid], lip_dict[train_valid]), axis=-1)
        aud_params['n_feat'] = aud_params['n_feat'] + lip_params['n_feat']


    # make sure ult_params has 'n_feat' in it for output vector size
    if 'window' not in aud_params.keys():
        aud_params['window'] = 5 if aud_dict['train'].ndim == 3 else None

    # build model
    if args.model_type == 'LSTM':
        model = build_LSTM_model(ult_params, aud_params)
    else:
        model = build_DNN_model(ult_params, aud_params)

    # compile model
    opt = Adam(learning_rate=args.lr)
    # add SSIM if ult is not dlc
    met = SSIMLoss if ult_params['n_feat'] == 64 * 128 else None
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=met)
    logging.info(f'Initial lr: {args.lr}.')

    # name model
    # want to save the name of the data used so that we know what to use for testing
    data_name = os.path.basename(ult_path)[3:-7]
    model_name = args.model_type + data_name
    model_path = os.path.join(args.model_dir, model_name)
    # train model
    train_model(model, model_path, ult_dict, aud_dict, args.epochs, args.batch, args.patience,
                early_stop=args.no_stop, profile=args.profile)
    # save model
    save_model(model, model_path)
    # eval on test set
    logging.info('Evaluating on test set')
    results = model.evaluate(aud_dict['test'], ult_dict['test'], batch_size=128, verbose=2, return_dict=True)
    for key in results:
        logging.info(f"{key}: {results[key]}")

if __name__ == "__main__":
    args = get_args()
    log_path = os.path.join(args.model_dir, 'training.log')
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # log args
    for arg, value in vars(args).items():
        logging.info(f"Argument {arg}: {value}\n")

    main(args)
