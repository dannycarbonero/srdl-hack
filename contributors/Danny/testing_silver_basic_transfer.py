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

LOO_subjects = generate_LOO_subjects()

with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

figure_directory ='figures/'
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


#%% process data for predicting

testing_data = data[data['subject'].isin(LOO_subjects)]
series = np.stack(np.array(testing_data.series))
time = np.stack(np.array(testing_data.time))
time = time - time[:,0].reshape(-1, 1)
classifications = np.array(testing_data.classification)

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


#%% predict
predictions = model.predict(np.expand_dims(series_downsampled, axis = 2).squeeze())
predictions = predictions.squeeze()


#%% assess predictions
prediction_peaks = []
for i in range(predictions.shape[0]):
    prediction_peaks.append(signal.find_peaks(predictions[i,:], height = height, width = width, distance = distance)[0])
paired_classifications, predictions_bin = refined_classification(prediction_peaks, classifications_bin, labels)


#%% plot
plt.figure()
confusion_matrix = metrics.confusion_matrix(paired_classifications, predictions_bin)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.title('Un-tuned Transfer Learning')
plt.savefig(figure_directory + 'un-tuned_transfer_LOO.png')
plt.show()

#%% THEIR DATA - check RippleNet_path/RippleNet_interactive_prototype.ipynb