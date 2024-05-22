import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import pickle
from sklearn import metrics

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})  # Set the default font size to 12

from contributors.Danny.initial_submission.utilities import build_data_sets, find_optimum_ROC_threshold, load_RippleNet, binarize_RippleNet, calculate_prediction_statistics, binarize_predictions

#%% function to return statistics, and ROC axis
def test_network(data, LOO_subjects, network_load_directory = None, Basic = False, LOO = False, Priors = False):

    # some constants
    silver_Fs = 2035  # from simulation q
    cut_factor = 0.75
    cut_points = int(silver_Fs * cut_factor)
    RippleNet_Fs = 1250
    label_center_s = 1
    pre_center_s = 0.1
    post_center_s = 0.05
    window_bounds = [label_center_s - pre_center_s, label_center_s + post_center_s]

    #%% initialize variables
    paired_classifications = []
    predictions_bin = []
    optimal_thresholds = []
    ROC_statistics = []
    confusion_matrices = []
    classifications = []
    event_probabilities = []
    labels = []
    predictions_aggregate = []
    statistics_th = []
    statistics_50 = []
    ROC_aucs = []

    if Basic:
        model = load_RippleNet('code')
        model = binarize_RippleNet(model)

    if Priors:
        model = keras.models.load_model(network_load_directory + 'RippleNet_tuned_priors.h5')
        model.summary()

        with open(network_load_directory + 'val_frame.pkl', 'rb') as file:
            validation_frame = pickle.load(file)

    for subject in LOO_subjects:

        if LOO:
            model = keras.models.load_model(network_load_directory + 'RippleNet_tuned_optimal_' + subject + '.h5')
            model.summary()

            with open(network_load_directory + subject + '_val_frame.pkl', 'rb') as file:
                validation_frame = pickle.load(file)

        _, validation_data = build_data_sets(validation_frame, cut_factor=cut_factor, silver_Fs=silver_Fs,
                                             RippleNet_Fs=RippleNet_Fs, label_center_s=label_center_s,
                                             pre_center_s=pre_center_s, post_center_s=post_center_s)
        predictions = model.predict(np.expand_dims(validation_data['series_downsampled'], axis=2)).squeeze()
        probabilities = predictions.copy()
        optimal_probability_threshold, optimal_operating_point = find_optimum_ROC_threshold(probabilities, validation_data[
            'classifications'])
        optimal_thresholds.append(optimal_probability_threshold)

        testing_frame = data.copy()[data['subject'] == subject]
        _, testing_data = build_data_sets(testing_frame, cut_factor=cut_factor, silver_Fs=silver_Fs,
                                          RippleNet_Fs=RippleNet_Fs, label_center_s=label_center_s,
                                          pre_center_s=pre_center_s, post_center_s=post_center_s)
        predictions = model.predict(np.expand_dims(testing_data['series_downsampled'], axis=2)).squeeze()
        probabilities = predictions.copy()
        event_probabilities.append(probabilities)
        predictions_aggregate.append(predictions)

        paired_classifications_working = testing_data['classifications']
        predictions_bin_working = binarize_predictions(probabilities.copy(), optimal_probability_threshold)

        statistics_th.append((calculate_prediction_statistics(paired_classifications_working, predictions_bin_working)))

        predictions_bin_50_working = binarize_predictions(probabilities.copy(), 0.5)
        statistics_50.append(calculate_prediction_statistics(paired_classifications_working, predictions_bin_50_working))

        ROC_statistics.append(metrics.roc_curve(testing_data['classifications'], probabilities))
        ROC_aucs.append(metrics.roc_auc_score(testing_data['classifications'], probabilities))

        confusion_matrices.append(
            metrics.confusion_matrix(paired_classifications_working, predictions_bin_working).ravel())  # tn, fp, fn, tp

        classifications.append(testing_data['classifications'])
        labels.append(testing_data['labels'])

        paired_classifications.append(paired_classifications_working)
        predictions_bin.append(predictions_bin_working)

