import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from directory_handling import get_parent_path
from utilities import binarize_classifications, make_refined_labels, create_training_subset, generate_LOO_subjects

#%% load Our Data
silver_Fs = 2035 # from simulation
data_path = get_parent_path('data', subdirectory ='Spike Ripples/silver')

with open(data_path + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

#%%
yes_frame = data[data['classification'] == 'y'].reset_index().drop('index', axis = 1)
testing_yes_indices = yes_frame.sample(200, replace = False).index
yes_frame_training = yes_frame.copy().drop(testing_yes_indices)
yes_frame_testing = yes_frame.copy().iloc[testing_yes_indices, :]

no_frame = data[data['classification'] == 'n'].reset_index().drop('index', axis = 1)
training_no_indices = no_frame.sample(n = int(yes_frame_training.shape[0]/2), replace=False).index
no_frame_training = no_frame.copy().iloc[training_no_indices, :]
no_frame_testing = no_frame.drop(training_no_indices)

bk_frame = data[data['classification'] == 'bk'].reset_index().drop('index', axis = 1)
training_bk_indices = bk_frame.sample(n = int(yes_frame_training.shape[0]/2), replace=False).index
bk_frame_training = bk_frame.copy().iloc[training_bk_indices, :]
bk_frame_testing = bk_frame.drop(training_bk_indices)

training_frame = pd.concat([yes_frame_training, no_frame_training, bk_frame_training])
testing_frame = pd.concat([yes_frame_testing, no_frame_testing, bk_frame_testing])

with open(data_path + 'sanity_check_training.pkl', 'wb') as file:
    pickle.dump(training_frame, file)

with open(data_path + 'sanity_check_testing.pkl', 'wb') as file:
    pickle.dump(testing_frame, file)


#%% a few constants
cut_fractor = 0.75
cut_points = int(silver_Fs * cut_fractor)
RippleNet_Fs = 1250
label_center_s = 1
pre_center_s = 0.1
post_center_s = 0.05

#%% training params
batch_size = 32
epochs = 256

network_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_sanity_check_' + str(epochs) + '_epochs/', make = True)

#%% train

#load their model
RippleNet_path = get_parent_path('scc', subdirectory='RippleNet')
model_file = RippleNet_path + 'best_model.pkl'
with open(model_file, 'rb') as f:
    best_model = pickle.load(f)
    print(best_model)

model = keras.models.load_model(RippleNet_path + best_model['model_file'])
model.summary()



model_checkpoint = tf.keras.callbacks.ModelCheckpoint(network_directory + 'RippleNet_tuned_optimal_sanity_check.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
checkpoint_history = keras.callbacks.CSVLogger(network_directory + 'RippleNet_tuning_history_sanity_check.csv')
checkpoint_list = [model_checkpoint, checkpoint_history]


# training data
training_series = np.stack(np.array(training_frame.series))[:, cut_points:-cut_points]
training_time = np.stack(np.array(training_frame.time))
training_time = training_time - training_time[:,0].reshape(-1,1)
training_time = training_time[:, cut_points:-cut_points]
training_classifications = np.array(training_frame.classification)
nan_series = np.where(np.any(np.isnan(training_series), axis=1))[0]
if np.any(nan_series):
    training_series = np.delete(training_series, nan_series, axis=0)
    training_time = np.delete(training_time, nan_series, axis=0)
    training_classifications = np.delete(training_classifications, nan_series, axis=0)
training_classifications_bin = binarize_classifications(training_classifications)

# validation data
# validation_series = np.stack(np.array(validation_frame.series))[:, cut_points:-cut_points]
# validation_time = np.stack(np.array(validation_frame.time))
# validation_time = validation_time - validation_time[:,0].reshape(-1,1)
# validation_time = validation_time[:, cut_points:-cut_points]
# validation_classifications = np.array(validation_frame.classification)
# nan_series = np.where(np.any(np.isnan(validation_series), axis=1))[0]
# if np.any(nan_series):
#     validation_series = np.delete(validation_series, nan_series, axis=0)
#     validation_time = np.delete(validation_time, nan_series, axis=0)
#     validation_classifications = np.delete(validation_classifications, nan_series, axis=0)
# validation_classifications_bin = binarize_classifications(validation_classifications)

# downsample
training_series_downsampled = np.expand_dims(signal.resample(training_series, int(RippleNet_Fs / silver_Fs * training_series.shape[1]), axis=1), 2)
training_time_downsampled = signal.resample(training_time, int(RippleNet_Fs / silver_Fs * training_time.shape[1]), axis = 1)
# validation_series_downsampled = np.expand_dims(signal.resample(validation_series, int(RippleNet_Fs / silver_Fs * validation_series.shape[1]), axis=1), 2)
# validation_time_downsampled = signal.resample(training_time, int(RippleNet_Fs / silver_Fs * validation_time.shape[1]), axis = 1)

# make labels
training_labels = np.expand_dims(make_refined_labels(training_classifications, training_time_downsampled, center_s = label_center_s, pre_center_s = pre_center_s, post_center_s = post_center_s), 2)
# validation_labels = np.expand_dims(make_refined_labels(validation_classifications, validation_time_downsampled, center_s = label_center_s, pre_center_s = pre_center_s, post_center_s = post_center_s), 2)

# create data sets
training_set = tf.data.Dataset.from_tensor_slices((training_series_downsampled, training_labels))
training_set = training_set.shuffle(training_series_downsampled.shape[0])
training_set = training_set.batch(batch_size)

# validation_set = tf.data.Dataset.from_tensor_slices((validation_series_downsampled, validation_labels))
# validation_set = validation_set.shuffle(validation_series_downsampled.shape[0])


history = model.fit(training_set, epochs = epochs, callbacks = checkpoint_list)#, validation_data=validation_set)
with open(network_directory + 'RippleNet_tuning_history_sanity_check.pkl', 'wb') as file:
    pickle.dump(history.history, file)

model.save(network_directory + 'RippleNet_tuned_sanity_check.h5')