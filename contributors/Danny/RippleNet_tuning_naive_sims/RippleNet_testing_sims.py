import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

#%% their model
# model_file = '/home/c/C.ode/RippleNet/best_model.pkl'
# # model_file = 'C:/Users/C/C.ode/RippleNet/best_model.pkl'
# with open(model_file, 'rb') as f:
#     best_model = pickle.load(f)
#     print(best_model)
#
# model = keras.models.load_model('/home/c/C.ode/RippleNet/' + best_model['model_file'])
# model.summary()

#%% our model

model = keras.models.load_model('RippleNet_tuned.h5')
model.summary()


#%% Our Data
Fs = 2035 # from simulation
# data_directory = '/home/SSD 1/Neural Data/Spike Ripples/'
data_directory = '/home/warehaus/Neural Data/Spike Ripples/naive_sim/'

spikes = np.array(pd.read_csv(data_directory + 'spikes_test.csv', header = None))
spike_labels = np.zeros(spikes.shape)
ripples = np.array(pd.read_csv(data_directory + 'ripples_test.csv', header = None))
ripple_labels = np.array(pd.read_csv(data_directory + 'labels_test.csv', header = None))


spikes_downsampled = signal.resample(spikes, int(1250/Fs * spikes.shape[1]), axis = 1)
ripples_downsampled = signal.resample(ripples, int(1250/Fs * ripples.shape[1]), axis  = 1)
labels_downsampled = signal.resample(ripple_labels, int(1250/Fs * ripples.shape[1]), axis  = 1)

spike_predictions = model.predict(np.expand_dims(spikes_downsampled, 2)).squeeze()
ripple_predictions = model.predict(np.expand_dims(ripples_downsampled, 2)).squeeze()

#%%
height = 0.5
width = 25
distance = 60
ripple_peaks = []
for i in range(ripple_predictions.shape[0]):
    ripple_peaks.append(signal.find_peaks(ripple_predictions[i,:], height = 0.5, width = 3, distance = 60)[0])

spike_peaks = []
for i in range(ripple_predictions.shape[0]):
    spike_peaks.append(signal.find_peaks(spike_predictions[i,:], height = 0.5, width = 3, distance = 60)[0])

#%%
ripple_label_ranges = []
for i in range(labels_downsampled.shape[0]):
    ripple_label_ranges.append(np.where(labels_downsampled[i,:] > 0.005))

ripple_predictions_bin = []
ripple_truth_bin = []
for i in range(len(ripple_peaks)):
    if np.any(ripple_peaks[i]):
        for j in range(len(ripple_peaks[i])):
            if np.any(ripple_label_ranges[i] == ripple_peaks[i][j]):
                ripple_predictions_bin.append(1)
                ripple_truth_bin.append(1)
            else:
                ripple_predictions_bin.append(1)
                ripple_truth_bin.append(0)
    else:
        ripple_predictions_bin.append(0)
        ripple_truth_bin.append(1)

spike_predictions_bin = []
spike_truth_bin = []
for i in range(len(spike_peaks)):
    if np.any(spike_peaks[i]):
        for j in range(len(spike_peaks[i])):
            spike_predictions_bin.append(1)
            spike_truth_bin.append(0)
    else:
        spike_predictions_bin.append(0)
        spike_truth_bin.append(0)


#%%
from sklearn import metrics

predictions = np.concatenate((ripple_predictions_bin, spike_predictions_bin))
truth = np.concatenate((ripple_truth_bin, spike_truth_bin))


confusion_matrix = metrics.confusion_matrix(truth, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#%% THEIR DATA
import os

Fs = 1250 # Hz, sampling freq
lag = int(100 * Fs / 1000) # 100 ms @ Fs

file_mode = 'r'
session = 'm4029_session1'  # holdout dataset
file_path = os.path.join('data', '{}.h5'.format(session))
f = h5py.File('/home/c/C.ode/RippleNet/' + file_path, file_mode)
print('opened file {} ({})'.format(file_path, f))

lfp = f[session]['lfp'][:]
segment_length = int(0.5 * Fs)  # Fs
# run predictions n times with shifts of length segment_length / n,
# then final output will be averaged
n = 5  # nicely divisible with Fs=1250
shift = int(segment_length / n)
container = []
for i in range(n):
    lfp_reshaped = np.concatenate((np.zeros((1, i * shift, 1)),
                                   np.expand_dims(np.expand_dims(lfp, 0), -1)), axis=1)

    # pad with zeros
    lfp_reshaped = np.concatenate((lfp_reshaped,
                                   np.zeros((1, segment_length -
                                             (lfp_reshaped.size % segment_length), 1))),
                                  axis=1)

    # reshape into segments of length
    lfp_reshaped = lfp_reshaped.reshape((-1, segment_length, 1))

    # run prediction on data
    y_hat = model.predict(lfp_reshaped)

    # Reshape to zero-padded size
    y_hat = y_hat.reshape((1, -1, 1))[:, :lfp_reshaped.size, :]

    # strip elements that were padded with zeros
    container.append(y_hat[:, i * shift:i * shift + lfp.size, :])

# average or median
y_hat = np.median(container, axis=0).flatten()

y_hat = y_hat.flatten()