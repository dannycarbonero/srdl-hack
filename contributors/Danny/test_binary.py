import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import h5py
import pickle
import pandas as pd
from scipy import signal
from sklearn import metrics


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})  # Set the default font size to 12


from directory_handling import get_parent_path
from utilities import generate_LOO_subjects, pull_event_probabilities, build_data_sets, find_optimum_ROC_threshold, load_RippleNet, binarize_RippleNet, calculate_prediction_statistics, binarize_predictions
from pathlib import Path

#%% load Our Data
silver_Fs = 2035 # from simulation q
data_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver')

with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

# network_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_LOO_128_epochs_binary_final')
network_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_tuned_priors_128_epochs_binary_final')
# network_directory = get_parent_path('data', subdirectory = 'Spike Ripples/silver/RippleNet_transfer_LOO_128_epochs_binary_final')

Basic = False
LOO = False
Priors = True



#%%
LOO_subjects = generate_LOO_subjects()


#%%
# some constants
cut_factor = 0.75
cut_points = int(silver_Fs * cut_factor)
RippleNet_Fs = 1250
label_center_s = 1
pre_center_s = 0.1
post_center_s = 0.05
window_bounds = [label_center_s - pre_center_s, label_center_s + post_center_s]

paired_classifications = []
predictions_bin = []
optimal_thresholds = []
ROC_statistics = []
confusion_matrices = []
classifications = []
event_probabilities = []
labels = []
predictions_aggregate = []

if Priors:
    model = keras.models.load_model(network_directory + 'RippleNet_tuned_priors.h5')
    model.summary()

    with open(network_directory + 'val_frame.pkl', 'rb') as file:
        validation_frame = pickle.load(file)

if Basic:
    model = load_RippleNet('code')
    model = binarize_RippleNet(model)

for subject in LOO_subjects:

    if LOO:
        model = keras.models.load_model(network_directory + 'RippleNet_tuned_' + subject + '.h5')
        model.summary()

        with open(network_directory + subject + '_val_frame.pkl', 'rb') as file:
            validation_frame = pickle.load(file)

    _, validation_data = build_data_sets(validation_frame,  cut_factor = cut_factor, silver_Fs = silver_Fs, RippleNet_Fs = RippleNet_Fs, label_center_s = label_center_s, pre_center_s = pre_center_s, post_center_s = post_center_s)
    predictions = model.predict(np.expand_dims(validation_data['series_downsampled'], axis = 2)).squeeze()
    probabilities = predictions.copy()
    optimal_probability_threshold, optimal_operating_point = find_optimum_ROC_threshold(probabilities, validation_data['classifications'])
    optimal_thresholds.append(optimal_probability_threshold)

    testing_frame = data.copy()[data['subject']==subject]
    _, testing_data = build_data_sets(testing_frame,  cut_factor = cut_factor, silver_Fs = silver_Fs, RippleNet_Fs = RippleNet_Fs, label_center_s = label_center_s, pre_center_s = pre_center_s, post_center_s = post_center_s)
    predictions = model.predict(np.expand_dims(testing_data['series_downsampled'], axis = 2)).squeeze()
    probabilities = predictions.copy()
    event_probabilities.append(probabilities)
    predictions_aggregate.append(predictions)

    paired_classifications_working = testing_data['classifications']
    predictions_bin_working = binarize_predictions(probabilities.copy(), optimal_probability_threshold)

    ROC_statistics.append(metrics.roc_curve(testing_data['classifications'], probabilities))



    confusion_matrices.append(metrics.confusion_matrix(paired_classifications_working, predictions_bin_working).ravel())  # tn, fp, fn, tp

    classifications.append(testing_data['classifications'])
    labels.append(testing_data['labels'])

    paired_classifications.append(paired_classifications_working)
    predictions_bin.append(predictions_bin_working)


#%% cummulative statistics
optimal_probability_threshold_cum, operating_point_cum = find_optimum_ROC_threshold(np.concatenate(event_probabilities),np.concatenate(classifications))
ROC_curve_cum = metrics.roc_curve(np.concatenate(classifications), np.concatenate(event_probabilities))
AUC_ROC_curve_cum = metrics.roc_auc_score(np.concatenate(classifications), np.concatenate(event_probabilities))
paired_classifications_cum = np.concatenate(classifications)
predictions_bin_cum = binarize_predictions(np.concatenate(predictions_aggregate),optimal_probability_threshold_cum)

paired_classifications_50 = paired_classifications_cum
predictions_bin_50 = binarize_predictions(np.concatenate(predictions_aggregate),0.5)

sens_agg, spec_agg, ppv_agg, npv_agg, accuracy_agg = calculate_prediction_statistics(np.concatenate(paired_classifications), np.concatenate(predictions_bin))
sens_opt, spec_opt, ppv_opt, npv_opt, accuracy_opt = calculate_prediction_statistics(paired_classifications_cum, predictions_bin_cum)
sens_50, spec_50, ppv_50, npv_50, accuracy_50 = calculate_prediction_statistics(paired_classifications_50, predictions_bin_50)
prediction_statistics = np.vstack(((sens_50, spec_50, ppv_50, npv_50, accuracy_50), (sens_agg, spec_agg, ppv_agg, npv_agg, accuracy_agg), (sens_opt, spec_opt, ppv_opt, npv_opt, accuracy_opt)))

#%% plotting

fig = plt.figure(figsize=(10/1.5, 8/1.5))

plt.rcParams['font.sans-serif'] = ['Arial']

# Adding ROC plot
ax_roc = plt.subplot2grid((4, 1), (0, 0), rowspan=4, fig=fig)  # Allocate 3/4 of the figure to ROC
for i in range(len(ROC_statistics)):
    ax_roc.plot(ROC_statistics[i][0], ROC_statistics[i][1], alpha=0.66)
ax_roc.plot(ROC_curve_cum[0], ROC_curve_cum[1], color='k')
ax_roc.plot(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100), color='k', linestyle='--')
ax_roc.set_ylim([-0.05, 1.05])
ax_roc.set_xlim([-0.05, 1.05])
ax_roc.scatter(operating_point_cum[0], operating_point_cum[1], color='r', s = 65)
ax_roc.set_xlabel('False Positive Rate', fontsize = 14)
ax_roc.set_ylabel('True Positive Rate', fontsize = 14)
ax_roc.legend(LOO_subjects)
ax_roc.spines[['right', 'top']].set_visible(False)
ax_roc.set_title(f'Priors Tuning, auc: {AUC_ROC_curve_cum:.4f}')

columns = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy']
rows = ['$p_{0.5}$', '$p_{validation}$', '$p_{opt}$']
formatted_prediction_statistics = [[f'{value:.4f}' for value in row] for row in prediction_statistics]

table_data = [columns]  # Header row
table_data.extend([[row_label] + row for row_label, row in zip(rows, formatted_prediction_statistics)])

import csv
csv_filename = "prediction_statistics_transfer_bin.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table_data)


# # Adding the table at the bottom, taking up 25% of the figure
# # ax_table = plt.subplot2grid((4, 1), (3, 0), fig=fig)
# ax_table.axis('tight')
# ax_table.axis('off')
# tbl = ax_table.table(cellText=formatted_prediction_statistics, rowLabels=rows, colLabels=columns, loc='center', cellLoc='center')
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(10)
# tbl.scale(1, 1.5)  # You may adjust these scaling factors as needed

plt.tight_layout()
fig.savefig('transfer_bin.svg')
fig.show()
