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
from utilities import load_RippleNet, binarize_RippleNet, freeze_RippleNet

RippleNet = load_RippleNet('code')
RippleNet_bin = binarize_RippleNet(RippleNet)