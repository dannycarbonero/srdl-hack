import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from directory_handling import get_parent_path
from utilities import binarize_classifications, make_labels, create_training_subset

#%% load their model
RippleNet_path = get_parent_path('code', subdirectory = 'RippleNet')
model_file = RippleNet_path + 'best_model.pkl'
with open(model_file, 'rb') as f:
    best_model = pickle.load(f)
    print(best_model)

model = keras.models.load_model(RippleNet_path + best_model['model_file'])
model.summary()


#%% load Our Data
silver_Fs = 2035 # from simulation
data_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver')

with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

#%% define LOO subjects
numbers = ['03', '07', '11', '15', '33', '43']
LOO_subjects = ["pBECTS0" + number for number in numbers]

#%% a few constants
cut_fractor = 0.75
cut_points = int(silver_Fs * cut_fractor)
RippleNet_Fs = 1250

#%% training params
batch_size = 32
epochs = 3

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('net_tuned_best.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
checkpoint_history = keras.callbacks.CSVLogger('RippleNet_tuning_history.csv')
checkpoint_list = [model_checkpoint, checkpoint_history]


#%% train
subject = LOO_subjects[0]
i = 0
#for i, subject in zip(range(len(LOO_subjects)), LOO_subjects):

print('Training on subject %i of %i' %(i, len(LOO_subjects)))

# pull data
validation_frame = data.copy()[data['subject'] == subject]
training_frame = data.copy()[data['subject']!=subject]
training_frame = create_training_subset(training_frame, int(training_frame['classification'].value_counts()['y'] * 2))

# training data
training_series = np.stack(np.array(training_frame.series))[:, cut_points:-cut_points]
training_classifications = np.array(training_frame.classification)
nan_series = np.where(np.any(np.isnan(training_series), axis=1))[0]
if np.any(nan_series):
    training_series = np.delete(training_series, nan_series, axis=0)
    training_classifications = np.delete(training_classifications, nan_series, axis=0)
training_classifications_bin = binarize_classifications(training_classifications)

# validation data
validation_series = np.stack(np.array(validation_frame.series))[:, cut_points:-cut_points]
validation_classifications = np.array(validation_frame.classification)
nan_series = np.where(np.any(np.isnan(validation_series), axis=1))[0]
if np.any(nan_series):
    validation_series = np.delete(validation_series, nan_series, axis=0)
    validation_classifications = np.delete(validation_classifications, nan_series, axis=0)
validation_classifications_bin = binarize_classifications(validation_classifications)

# downsample
training_series_downsampled = np.expand_dims(signal.resample(training_series, int(RippleNet_Fs / silver_Fs * training_series.shape[1]), axis=1), 2)
validation_series_downsampled = np.expand_dims(signal.resample(validation_series, int(RippleNet_Fs / silver_Fs * validation_series.shape[1]), axis=1), 2)
# make labels
training_labels = np.expand_dims(make_labels(training_classifications, training_series_downsampled.shape[1]), 2)
validation_labels = np.expand_dims(make_labels(validation_classifications, validation_series_downsampled.shape[1]), 2)

# create data sets
training_set = tf.data.Dataset.from_tensor_slices((training_series_downsampled, training_labels))
training_set = training_set.shuffle(training_series_downsampled.shape[0])
training_set = training_set.batch(batch_size)

validation_set = tf.data.Dataset.from_tensor_slices((validation_series_downsampled, validation_labels))
validation_set = validation_set.shuffle(validation_series_downsampled.shape[0])


history = model.fit(training_set, epochs=epochs, callbacks=checkpoint_list, validation_data=validation_set)
#    with open('RippleNet_tuning_history_' + subject + '.pkl', 'wb') as file:
#        pickle.dump(history.history, file)

model.save('net_tuned_manual_save.h5')

