import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

model_file = '/best_model.pkl'
# model_file = 'C:/Users/C/C.ode/RippleNet/best_model.pkl'
with open(model_file, 'rb') as f:
    best_model = pickle.load(f)
    print(best_model)

model = keras.models.load_model('/home/c/C.ode/RippleNet/' + best_model['model_file'])


#%%
Fs = 2035
data_directory = '/home/SSD 1/Neural Data/Spike Ripples/'
spikes = []
ripples = []
labels = []

for i in range(10):
    if i == 0:
        validation_spikes = np.array(pd.read_csv(data_directory + 'spikes_' + str(i) + '.csv', header = None))
        validation_ripples = np.array(pd.read_csv(data_directory + 'ripples_' + str(i) + '.csv', header = None))
        validation_labels = np.array(pd.read_csv(data_directory + 'labels_' + str(i) + '.csv', header = None))
    else:
        spikes.append(np.array(pd.read_csv(data_directory + 'spikes_' + str(i) + '.csv', header=None)))
        ripples.append(np.array(pd.read_csv(data_directory + 'ripples_' + str(i) + '.csv', header=None)))
        labels.append(np.array(pd.read_csv(data_directory + 'labels_' + str(i) + '.csv', header=None)))

spikes = np.vstack(spikes)
ripples = np.vstack(ripples)
time_series = np.vstack((spikes, ripples))

data_labels = np.vstack(labels)
spike_labels = np.zeros(spikes.shape)
data_labels_master = np.vstack((spike_labels, data_labels))

time_series_val = np.vstack((np.vstack(validation_spikes), np.vstack(validation_ripples)))
spike_labels_val = np.zeros(validation_spikes.shape)
data_labels_master_val = np.vstack((spike_labels_val, np.vstack(validation_labels)))



#%%

time_series_downsampled = signal.resample(time_series, int(1250/Fs * spikes.shape[1]), axis = 1)
data_labels_master_downsampled = signal.resample(data_labels_master, int(1250/Fs * spikes.shape[1]), axis = 1)

time_series_val_downsampled = signal.resample(time_series_val, int(1250/Fs * spikes.shape[1]), axis = 1)
data_labels_master_val_downsampled = signal.resample(data_labels_master_val, int(1250/Fs * spikes.shape[1]), axis = 1)

data_labels_master_downsampled[data_labels_master_downsampled > 0.005] = 1
data_labels_master_val_downsampled[data_labels_master_val_downsampled > 0.005] = 1

data_labels_master_downsampled[data_labels_master_downsampled <= 0.005] = 0
data_labels_master_val_downsampled[data_labels_master_val_downsampled <= 0.005] = 0

time_series_downsampled = np.expand_dims(time_series_downsampled, 2)
time_series_val_downsampled = np.expand_dims(time_series_val_downsampled, 2)

data_labels_master_downsampled = np.expand_dims(data_labels_master_downsampled, 2)
data_labels_master_val_downsampled = np.expand_dims(data_labels_master_val_downsampled, 2)

#%%
batch_size = 32
epochs = 10

data_set = tf.data.Dataset.from_tensor_slices((time_series_downsampled, data_labels_master_downsampled))
data_set = data_set.shuffle(time_series_downsampled.shape[0])
data_set = data_set.batch(batch_size)

val_set = tf.data.Dataset.from_tensor_slices((time_series_val_downsampled, data_labels_master_val_downsampled))
val_set = val_set.shuffle(time_series_val_downsampled.shape[0])


#%%

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('RippleNet_tuned.h5', monitor = 'mse', verbose = 1, save_best_only = True, mode = 'min')
checkpoint_history = keras.callbacks.CSVLogger('RippleNet_tuning_history.csv')
checkpoint_list = [model_checkpoint, checkpoint_history]

#%%

history = model.fit(data_set, epochs = epochs, callbacks = checkpoint_list, validation_data = val_set)

with open('RippleNet_tuning_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

model.save('RippleNet_tuned.h5')
