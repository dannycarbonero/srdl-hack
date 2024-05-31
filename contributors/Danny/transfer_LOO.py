import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
from scipy import signal

from directory_handling import get_parent_path
from utilities import binarize_classifications, create_training_subset, generate_LOO_subjects

#%% load Our Data
silver_Fs = 2035 # from simulation
data_path = get_parent_path('data', subdirectory ='Spike Ripples/silver')

with open(data_path + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

#%% define LOO subjects
LOO_subjects = generate_LOO_subjects()

#%% a few constants
cut_factor = 0.75
cut_points = int(silver_Fs * cut_factor)
RippleNet_Fs = 1250
label_center_s = 1
pre_center_s = 0.1
post_center_s = 0.05

#%% training params
batch_size = 32
epochs = 128
num_synthetic_ripples = [4000, 6000, 8000, 10000]

for num_ripples in num_synthetic_ripples:

    network_load_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_priors_' + str(128) + '_epochs_' + str(num_ripples) + '_SEs_binary', make = True)
    network_save_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_transfer_LOO_' + str(epochs) + '_epochs_' + str(num_ripples) + '_SEs_binary', make = True)

    for i, subject in zip(range(len(LOO_subjects)), LOO_subjects):

        model = keras.models.load_model(network_load_directory + 'RippleNet_tuned_priors.h5')

        print('Training on subject %i of %i' % (i + 1, len(LOO_subjects)))

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            network_save_directory + 'RippleNet_tuned_optimal_' + subject + '.h5', monitor='loss', verbose=1,
            save_best_only=True, mode='min')
        checkpoint_history = keras.callbacks.CSVLogger(
            network_save_directory + 'RippleNet_tuning_history_' + subject + '.csv')
        checkpoint_list = [model_checkpoint, checkpoint_history]

        # pull validation data
        LOO_frame = data.copy()[data['subject'] != subject]
        frame_y = LOO_frame[LOO_frame['classification'] == 'y']
        validation_frame = pd.concat((LOO_frame[LOO_frame['classification'] == 'y'].sample(int(frame_y.shape[0] * 0.1)),
                                      (LOO_frame[LOO_frame['classification'] == 'n'].sample(
                                          int(frame_y.shape[0] * 0.05))), (
                                          LOO_frame[LOO_frame['classification'] == 'bk'].sample(
                                              int(frame_y.shape[0] * 0.05)))))
        training_frame = LOO_frame.loc[LOO_frame.index.difference(validation_frame.index)]

        training_frame = create_training_subset(training_frame,
                                                int(training_frame['classification'].value_counts()['y'] * 2))

        print(training_frame['classification'].value_counts())

        with open(network_save_directory + subject + '_val_frame.pkl', 'wb') as file:
            pickle.dump(validation_frame, file)

        # training data processing
        training_series = np.stack(np.array(training_frame.series))[:, cut_points:-cut_points]
        training_time = np.stack(np.array(training_frame.time))
        training_time = training_time - training_time[:, 0].reshape(-1, 1)
        training_time = training_time[:, cut_points:-cut_points]
        training_classifications = np.array(training_frame.classification)
        nan_series = np.where(np.any(np.isnan(training_series), axis=1))[0]
        if np.any(nan_series):
            training_series = np.delete(training_series, nan_series, axis=0)
            training_time = np.delete(training_time, nan_series, axis=0)
            training_classifications = np.delete(training_classifications, nan_series, axis=0)
        training_classifications_bin = binarize_classifications(training_classifications)

        # validation data processing
        validation_series = np.stack(np.array(validation_frame.series))[:, cut_points:-cut_points]
        validation_time = np.stack(np.array(validation_frame.time))
        validation_time = validation_time - validation_time[:, 0].reshape(-1, 1)
        validation_time = validation_time[:, cut_points:-cut_points]
        validation_classifications = np.array(validation_frame.classification)
        nan_series = np.where(np.any(np.isnan(validation_series), axis=1))[0]
        if np.any(nan_series):
            validation_series = np.delete(validation_series, nan_series, axis=0)
            validation_time = np.delete(validation_time, nan_series, axis=0)
            validation_classifications = np.delete(validation_classifications, nan_series, axis=0)
        validation_classifications_bin = binarize_classifications(validation_classifications)

        # downsample
        training_series_downsampled = np.expand_dims(signal.resample(training_series, int(RippleNet_Fs / silver_Fs * training_series.shape[1]), axis=1), 2)
        training_time_downsampled = signal.resample(training_time,int(RippleNet_Fs / silver_Fs * training_time.shape[1]), axis=1)
        validation_series_downsampled = np.expand_dims(signal.resample(validation_series, int(RippleNet_Fs / silver_Fs * validation_series.shape[1]), axis=1), 2)
        validation_time_downsampled = signal.resample(training_time,int(RippleNet_Fs / silver_Fs * validation_time.shape[1]), axis=1)

        # get training labels
        training_labels = np.array(training_classifications_bin).reshape(-1, 1)
        validation_labels = np.array(validation_classifications_bin).reshape(-1, 1)

        # create data sets
        training_set = tf.data.Dataset.from_tensor_slices((training_series_downsampled, training_labels))
        training_set = training_set.shuffle(training_series_downsampled.shape[0])
        training_set = training_set.batch(batch_size)

        validation_set = tf.data.Dataset.from_tensor_slices((validation_series_downsampled, validation_labels))
        validation_set = validation_set.shuffle(validation_series_downsampled.shape[0]).batch(batch_size)

        history = model.fit(training_set, epochs=epochs, callbacks=checkpoint_list, validation_data=validation_set)
        with open(network_save_directory + 'RippleNet_tuning_history_' + subject + '.pkl', 'wb') as file:
            pickle.dump(history.history, file)

        model.save(network_save_directory + 'RippleNet_tuned_' + subject + '.h5')