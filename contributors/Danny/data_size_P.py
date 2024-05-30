import numpy as np
import pickle
import pandas as pd

from directory_handling import get_parent_path
from utilities import binarize_classifications, create_training_subset, generate_LOO_subjects, load_RippleNet, freeze_RippleNet, binarize_RippleNet

#%% load Our Data
silver_Fs = 2035 # from simulation
data_path = get_parent_path('data', subdirectory ='Spike Ripples/silver')

with open(data_path + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

print(data.shape)