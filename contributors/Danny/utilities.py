import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from scipy import signal

from directory_handling import get_parent_path

def build_data_sets(training_frame,  batch_size, validation_frame = None, cut_factor = 0.75, silver_Fs = 2035, RippleNet_Fs = 1250, label_center_s = 1, pre_center_s = 0.1, post_center_s = 0.05):

    cut_points = int(silver_Fs * cut_factor)

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
    training_time_downsampled = signal.resample(training_time, int(RippleNet_Fs / silver_Fs * training_time.shape[1]), axis=1)
    validation_series_downsampled = np.expand_dims(signal.resample(validation_series, int(RippleNet_Fs / silver_Fs * validation_series.shape[1]), axis=1), 2)
    validation_time_downsampled = signal.resample(training_time, int(RippleNet_Fs / silver_Fs * validation_time.shape[1]), axis=1)

    # make labels
    training_labels = np.expand_dims(make_refined_labels(training_classifications, training_time_downsampled, center_s=label_center_s, pre_center_s=pre_center_s, post_center_s=post_center_s), 2)
    validation_labels = np.expand_dims(make_refined_labels(validation_classifications, validation_time_downsampled, center_s=label_center_s, pre_center_s=pre_center_s, post_center_s=post_center_s), 2)

    # create data sets
    training_set = tf.data.Dataset.from_tensor_slices((training_series_downsampled, training_labels))
    training_set = training_set.shuffle(training_series_downsampled.shape[0])
    training_set = training_set.batch(batch_size)

    validation_set = tf.data.Dataset.from_tensor_slices((validation_series_downsampled, validation_labels))
    validation_set = validation_set.shuffle(validation_series_downsampled.shape[0])

    return training_set, validation_set



def binarize_classifications(classifications):

    classifications_bin = []
    for i in range(len(classifications)):
        if classifications[i] == 'y':
            classifications_bin.append(1)
        else:
            classifications_bin.append(0)

    return classifications_bin



def make_naive_labels(classifications, length):

    labels = []
    for i in range(len(classifications)):
        if classifications[i] == 'y':
            labels.append(np.ones((1,length)))
        else:
            labels.append(np.zeros((1, length)))

    labels = np.stack(labels).squeeze()

    return labels



def naive_classification(prediction_peaks, classifications_bin):

    paired_classifications = []
    predictions_bin = []

    for i in range(len(prediction_peaks)):

        if np.any(prediction_peaks[i]):

            # for j in range(len(prediction_peaks[i])):

            predictions_bin.append(1)
            paired_classifications.append(classifications_bin[i])

        else:
            predictions_bin.append(0)
            paired_classifications.append(classifications_bin[i])

    return paired_classifications, predictions_bin



def make_refined_labels(classifications, time, center_s = 1, pre_center_s = 0.1 , post_center_s = 0.05):

    labels = []
    for i in range(len(classifications)):
        if classifications[i] == 'y':
            labels.append((time[i,:] < (center_s + post_center_s)) & (time[i,:] > (center_s - pre_center_s)).astype('int'))
        else:
            labels.append(np.zeros(time.shape[1]))

    labels = np.stack(labels).squeeze()

    return labels



def refined_classification(prediction_peaks, classifications_bin, labels):

    paired_classifications = []
    predictions_bin = []

    for i in range(len(prediction_peaks)):

        if np.any(prediction_peaks[i]):

            for j in range(len(prediction_peaks[i])):

                if classifications_bin[i] == 0:
                    predictions_bin.append(1)
                    paired_classifications.append(classifications_bin[i])
                else:
                    # structure for checking if a peak is in the label
                    if labels[i,prediction_peaks[i][j]] == True:
                        predictions_bin.append(1)
                        paired_classifications.append(1)
                    else:
                        predictions_bin.append(1)
                        paired_classifications.append(0)

        else:
            predictions_bin.append(0)
            paired_classifications.append(classifications_bin[i])

    return paired_classifications, predictions_bin



def create_training_subset(training_frame, num_samples):

    num_y = num_samples // 2  # 50% of N
    num_n = num_samples // 4  # 25% of N
    num_bk = num_samples - num_y - num_n  # Remaining 25%

    # Filter the DataFrame based on classification
    df_y = training_frame[training_frame['classification'] == 'y'].sample(n=num_y)
    df_n = training_frame[training_frame['classification'] == 'n'].sample(n=num_n, replace=True)
    df_bk = training_frame[training_frame['classification'] == 'bk'].sample(n=num_bk, replace=True)

    # Concatenate the subsets
    subset_frame = pd.concat([df_y, df_n, df_bk])

    return subset_frame



def find_dataframe_overlap(df1, df2, m, n):
    """
    Iterate through df1 and find rows in df2 where columns 0-4 match.

    :param df1: First DataFrame to iterate through.
    :param df2: Second DataFrame to search for matching rows.
    :return: List of indices in df2 where rows match those in df1 based on the first five columns.
    """
    matching_indices = []

    # Convert the first five columns of each DataFrame to a more easily comparable format
    df1_str = df1.iloc[:, 0:5].astype(str).agg('-'.join, axis=1)
    df2_str = df2.iloc[:, 0:5].astype(str).agg('-'.join, axis=1)

    # Iterate through df1
    for i, row_val in enumerate(df1_str):
        # Find matches in df2
        matches = df2_str[df2_str == row_val].index.tolist()
        # matching_indices.extend(matches)
        # Sample the first m of every n indices from matches and extend matching_indices with these values
        for j in range(0, len(matches), n):
            # This ensures we add only the first m elements of the matches for each n-interval
            matching_indices.extend(matches[j:j + m])

    return matching_indices



def generate_LOO_subjects(numbers = None):
    numbers = ['03', '07', '11', '15', '33', '43']
    LOO_subjects = ["pBECTS0" + number for number in numbers]

    return LOO_subjects



def load_RippleNet(context):

    RippleNet_path = get_parent_path(context, subdirectory='RippleNet')
    model_file = RippleNet_path + 'best_model.pkl'
    with open(model_file, 'rb') as f:
        best_model = pkl.load(f)
        print(best_model)

    model = keras.models.load_model(RippleNet_path + best_model['model_file'])
    # model.summary()

    return model



def pull_event_probabilities():
    pass