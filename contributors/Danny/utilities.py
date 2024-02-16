import numpy as np
import pandas as pd
import pickle as pkl
from tensorflow import keras
from directory_handling import get_parent_path

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