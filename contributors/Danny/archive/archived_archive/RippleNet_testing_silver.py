import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from directory_handling import get_parent_path

def binarize_classifications(classifications):

    classifications_bin = []
    for i in range(len(classifications)):
        if classifications[i] == 'y':
            classifications_bin.append(1)
        else:
            classifications_bin.append(0)

    return classifications_bin


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


#%% pull data
series = np.stack(np.array(data.series))
labels = np.stack(np.array(data.log_Y))
classifications = np.array(data.classification)

#%% pull time
time = np.stack(np.array(data.time))
start_time_zeroed = np.array(data.time_start) - time[:,0]
time_window = np.array(data.time_stop) - np.array(data.time_start)
end_time_zeroed = np.array(start_time_zeroed + time_window)
event_times = np.stack((start_time_zeroed, end_time_zeroed)).T
for i in range(time.shape[0]):
    time[i,:] = time[i,:] - time[i,0]

#%% cut 1.25 seconds on each side, and remove traces with nan
cut_fractor = 0.75
cut_points = int(silver_Fs * cut_fractor)
series = series[:, cut_points:-cut_points]
labels = labels[:, cut_points:-cut_points]
time = time[:, cut_points:-cut_points]


nan_series = np.where(np.any(np.isnan(series), axis=1))[0]
# labels = np.delete(labels, nan_series, axis = 0)
# series = np.delete(series, nan_series, axis = 0)
# time = np.delete(time, nan_series, axis = 0)
# event_times = np.delete(event_times, nan_series, axis = 0)
classifications = np.delete(classifications, nan_series, axis = 0)
classifications_bin = binarize_classifications(classifications)


#%% downsample data
RippleNet_Fs = 1250

series_downsampled = signal.resample(series, int(RippleNet_Fs/silver_Fs * series.shape[1]), axis = 1)
labels_downsampled = signal.resample(labels, int(RippleNet_Fs/silver_Fs * labels.shape[1]), axis = 1)
time_downsampled = signal.resample(time, int(RippleNet_Fs/silver_Fs * labels.shape[1]), axis = 1)


#%%
classifications_bin = np.array(classifications_bin)
class_true = np.where(classifications_bin == 1)[0]
isY_true = np.where(labels_downsampled[:,0] !=0)[0]
incorrect_isYs = np.setxor1d(isY_true, class_true)


#%% predict
predictions = model.predict(np.expand_dims(series_downsampled, axis = 2).squeeze())
predictions = predictions.squeeze()





#%%
height = 0.5
width = 25
distance = 60
prediction_peaks = []

for i in range(predictions.shape[0]):
    prediction_peaks.append(signal.find_peaks(predictions[i,:], height = height, width = width, distance = distance)[0])


#%%
predictions = []
paired_classifications = []
for i in range(len(prediction_peaks)):

    if np.any(prediction_peaks[i]):

        for j in range(len(prediction_peaks[i])):

            if classifications_bin[i] == 0:
                predictions.append(1)
                paired_classifications.append(classifications_bin[i])
            else:

                if prediction_peaks[i][j]:
                    pass

    else:
        predictions.append(0)
        paired_classifications.append(classifications_bin[i])




#%%
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(truth, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#%% THEIR DATA - check RippleNet_path/RippleNet_interactive_prototype.ipynb