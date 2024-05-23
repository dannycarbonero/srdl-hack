import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import pickle
from sklearn import metrics

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})  # Set the default font size to 12

from directory_handling import get_parent_path
from utilities import build_data_sets, find_optimum_ROC_threshold, load_RippleNet, binarize_RippleNet, calculate_prediction_statistics, binarize_predictions, generate_LOO_subjects

#%% function to return statistics, and ROC axis
def test_network(data, LOO_subjects, network_load_directory = None, Basic = False, LOO = False, Priors = False, fig = None, subplot_dimensions = None, i = None):

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
    predictions_bin = []
    paired_classifications = []
    optimal_thresholds = []
    ROC_statistics = []
    ROC_aucs = []
    confusion_matrices = []
    classifications = []
    event_probabilities = []
    labels = []
    predictions_aggregate = []
    statistics_th = []
    statistics_50 = []

    if Basic:
        model = load_RippleNet('code')
        model = binarize_RippleNet(model)

    if Priors:
        model = keras.models.load_model(network_load_directory + 'RippleNet_tuned_priors.h5')

        with open(network_load_directory + 'val_frame.pkl', 'rb') as file:
            validation_frame = pickle.load(file)

    for subject in LOO_subjects:

        if Basic:
            validation_frame = data.copy()[data['subject'] != subject]

        if LOO:
            model = keras.models.load_model(network_load_directory + 'RippleNet_tuned_optimal_' + subject + '.h5')

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


    statistics_50 = np.vstack(statistics_50)
    statistics_th = np.vstack(statistics_th)

    optimal_probability_threshold_cum, operating_point_cum = find_optimum_ROC_threshold(np.concatenate(event_probabilities), np.concatenate(classifications))
    ROC_curve_cum = metrics.roc_curve(np.concatenate(classifications), np.concatenate(event_probabilities))
    AUC_ROC_curve_cum = metrics.roc_auc_score(np.concatenate(classifications), np.concatenate(event_probabilities))

    if fig:
        ax_roc = fig.add_subplot(subplot_dimensions[0], subplot_dimensions[1], i+1)
        for j in range(len(ROC_statistics)):
            ax_roc.plot(ROC_statistics[j][0], ROC_statistics[j][1], alpha=0.66)
        ax_roc.plot(ROC_curve_cum[0], ROC_curve_cum[1], color='k')
        ax_roc.plot(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100), color='k', linestyle='--')
        ax_roc.set_ylim([-0.05, 1.05])
        ax_roc.set_xlim([-0.05, 1.05])
        # ax_roc.scatter(operating_point_cum[0], operating_point_cum[1], color='r', s = 65)
        ax_roc.set_xlabel('False Positive Rate', fontsize=14)
        ax_roc.set_ylabel('True Positive Rate', fontsize=14)
        ax_roc.spines[['right', 'top']].set_visible(False)
        if i+1 == 0:
            ax_roc.legend({})
            ax_roc.legend(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Combined'])
        if i == 0:
            ax_roc.set_title('No tuning')
        elif i == 1:
            ax_roc.set_title(r'$\it{In\ vivo}$ data')
        elif i == 2:
            ax_roc.set_title('Synthetic Data Alone')
        elif i == 3:
            ax_roc.set_title(r'Synthetic + $\it{in\ vivo}$ data')

    return statistics_50, statistics_th, ROC_statistics, ROC_aucs, fig



#%%

data_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver')

with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

LOO_subjects = generate_LOO_subjects()

network_directories  = [get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_LOO_128_epochs_binary_final'),
                        get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_priors_128_epochs_binary_final'),
                        get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_transfer_LOO_128_epochs_binary_final')]

fig = plt.figure(figsize = (12,8))
variables = []
variables.append(test_network(data, LOO_subjects, Basic = True, fig = fig, subplot_dimensions = (2,2), i = 0))
variables.append(test_network(data, LOO_subjects, LOO = True, fig = variables[0][4], subplot_dimensions = (2,2), i = 1, network_load_directory = network_directories[0]))
variables.append(test_network(data, LOO_subjects, Priors = True, fig = variables[1][4], subplot_dimensions = (2,2), i = 2, network_load_directory = network_directories[1]))
variables.append(test_network(data, LOO_subjects, LOO = True, fig = variables[2][4], subplot_dimensions = (2,2), i = 3, network_load_directory = network_directories[2]))
variables[-1][4].tight_layout()
variables[-1][4].show()