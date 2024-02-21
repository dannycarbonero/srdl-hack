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
from sklearn import metrics

from directory_handling import get_parent_path
from utilities import binarize_classifications, refined_classification, make_refined_labels, generate_LOO_subjects
from pathlib import Path


#%% load Our Data
silver_Fs = 2035 # from simulation
data_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver')

with open(data_directory + 'sanity_check_testing.pkl', 'rb') as file:
    data = pickle.load(file)

network_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_sanity_check_128_epochs')
figure_directory ='figures/sanity_check_tuning_128_epochs/'
Path(figure_directory).mkdir(exist_ok = True)


#%%
# some constants
cut_fractor = 0.75
cut_points = int(silver_Fs * cut_fractor)
RippleNet_Fs = 1250
label_center_s = 1
pre_center_s = 0.1
post_center_s = 0.05

# prediction params
height = 0.75
width_s = .025
distance_s = .1
width = int(RippleNet_Fs * width_s)
distance = int(RippleNet_Fs * distance_s)

paired_classifications = []
predictions_bin = []

model = keras.models.load_model(network_directory + 'RippleNet_tuned_sanity_check.h5')
model.summary()

data_frame = data.copy()

series = np.stack(np.array(data_frame.series))
time = np.stack(np.array(data_frame.time))
time = time - time[:,0].reshape(-1, 1)
classifications = np.array(data_frame.classification)

series = series[:, cut_points:-cut_points]
time = time[:, cut_points:-cut_points]
nan_series = np.where(np.any(np.isnan(series), axis=1))[0]
if np.any(nan_series):
    series = np.delete(series, nan_series, axis = 0)
    time = np.delete(series, nan_series, axis = 0)
    classifications = np.delete(classifications, nan_series, axis = 0)
classifications_bin = binarize_classifications(classifications)

series_downsampled = signal.resample(series, int(RippleNet_Fs/silver_Fs * series.shape[1]), axis = 1)
time_downsampled = signal.resample(time, int(RippleNet_Fs/silver_Fs * series.shape[1]), axis = 1)
labels = make_refined_labels(classifications, time_downsampled, center_s = label_center_s, pre_center_s = pre_center_s, post_center_s = post_center_s)

predictions = model.predict(np.expand_dims(series_downsampled, axis = 2).squeeze())
predictions = predictions.squeeze()

prediction_peaks = []
for i in range(predictions.shape[0]):
    prediction_peaks.append(signal.find_peaks(predictions[i,:], height = height, width = width, distance = distance)[0])

paired_classifications, predictions_bin = refined_classification(prediction_peaks, classifications_bin, labels)

plt.figure()
confusion_matrix = metrics.confusion_matrix(paired_classifications, predictions_bin)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.title('')
plt.savefig(figure_directory + 'aggregate' + '.png')
plt.show()



#%% THEIR DATA - check RippleNet_path/RippleNet_interactive_prototype.ipynb