import numpy as np
import os
import h5py
from contributors.Danny.initial_submission.utilities import get_parent_path

mouse = True
rat = True
dataset_index = 0
ripple_path =  get_parent_path('code', 'RippleNet')


if mouse:
    # training and validation files
    f_name_train = 'train_{:02}.h5'
    f_name_val = 'validation_{:02}.h5'

    # training data
    f = h5py.File(os.path.join(ripple_path, 'data',
                               f_name_train.format(dataset_index)),
                  'r')
    X_train = np.expand_dims(f['X0'][:], -1)
    Y_train = f['Y'][:]
    f.close()

    # validation data
    f = h5py.File(os.path.join(ripple_path, 'data',
                               f_name_val.format(dataset_index)),
                  'r')
    X_val = np.expand_dims(f['X0'][:], -1)
    Y_val = f['Y'][:]
    f.close()

    # load some data for plotting
    f = h5py.File(os.path.join(ripple_path, 'data',
                               f_name_val.format(dataset_index)), 'r')
    X0 = f['X0'][:]
    X1 = f['X1'][:]
    S = f['S'][:]
    Y = f['Y'][:]
    S_freqs = f['S_freqs'][:]
    f.close()
#%%
# Add rat training/validation data to sets
if rat and mouse:
    # rat
    f_name_train = 'train_tingley_{:02}.h5'
    f_name_val = 'validation_tingley_{:02}.h5'

    # training data
    f = h5py.File(os.path.join(ripple_path, 'data',
                            f_name_train.format(dataset_index)),
                'r')
    X_train = np.concatenate((X_train, np.expand_dims(f['X0'][:], -1)))
    Y_train = np.concatenate((Y_train, f['Y'][:]))
    f.close()

    # validation data
    f = h5py.File(os.path.join(ripple_path, 'data',
                            f_name_val.format(dataset_index)),
                'r')
    X_val = np.concatenate((X_val, np.expand_dims(f['X0'][:], -1)))
    Y_val = np.concatenate((Y_val, f['Y'][:]))
    f.close()

    # load some data for plotting
    f = h5py.File(os.path.join(ripple_path, 'data',
                            f_name_val.format(dataset_index)), 'r')
    X0 = np.concatenate((X0, f['X0'][:]))
    X1 = np.concatenate((X1, f['X1'][:]))
    S = np.concatenate((S, f['S'][:]))
    Y = np.concatenate((Y, f['Y'][:]))
    f.close()
