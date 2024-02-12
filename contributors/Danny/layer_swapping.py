import sys

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

from directory_handling import get_parent_path
from utilities import binarize_classifications, make_refined_labels, create_training_subset, find_dataframe_overlap, generate_LOO_subjects

RippleNet_path = get_parent_path('code', subdirectory='RippleNet')
model_file = RippleNet_path + 'best_model.pkl'
with open(model_file, 'rb') as f:
    best_model = pickle.load(f)
    print(best_model)

model = keras.models.load_model(RippleNet_path + best_model['model_file'])