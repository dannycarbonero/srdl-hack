# CNN_Spectrogram_Algorithm

The CNN spike ripple detector: a method to classify spectrograms from EEG data using a convolutional neural network (CNN).

----

## Usage

See folder [Demo-Application](./Demo-Application) for an example applciation of the trained CNN spike ripple detector to simulated EEG data.

See folder [Demo-Training](./Demo-Training) for an example of how to train the CNN spike ripple detector using simulated spectrogram images.

Code in folder `fastai_v1` comes from fastai version 0.7 by Jeremy Howard: https://www.fast.ai/

## Data Structure

To run either demonstration, you must have a `data` folder of the following structure:

data/

├── train/

    ├── Yes
	
    ├── No
	
├── valid/

    ├── Yes
	
    ├── No
	
├── test/

For training, the `Yes` and `No` subfolders contain positive and negative case images on which we train the model. The `test` folder contains uncategorized images on which we test the model.

For application, the `test` folder contains new test data to be evaluated by the pretrained model (`full_trained_model.pkl`). For the code to run with this library, the `Yes` and `No` subfolders of `train` and `valid` cannot be empty: fill them with a few images from your test data -- this will not affect the output.

## Environment

Below is a step-by-step method to prepare an environment capable of running the notebooks:

0. Ensure you have both conda and pip installed

1. In terminal, load in a virtual environment with conda, give it a name (`cnn-specgram`):

`conda env create -f new_enviro.yml -n cnn-specgram`

`conda activate cnn-specgram`

3. Open the jupyter console to run notebooks:

`jupyter lab` 

4. When done, use `conda deactivate` to deactivate your virtual environment. To reload this environment in the future, use `conda activate cnn-specgram`, skipping step 2.

## Troubleshooting

<details>
  <summary>The trained model is only a 3-line text file</summary>

  Github limits the size of tracked files. You can use an extension to track larger files (https://git-lfs.com/); without this, you will just see a reference to file. 

Download the model using `wget`:
  ```
wget https://github.com/eschlaf2/CNN_Spectrogram_Algorithm/blob/master/full_trained_model.pkl
  ```

Initialize git LFS (on the SCC - you may need to install it on your home computer first; see https://git-lfs.com/):
```
git lfs install
```

For more info, see https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage.

  
</details>

<details>
  <summary>You get an error in bcolz because it uses deprecated np.float type</summary>

  If you replace all instances of `np.float` with `float` in */projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/toplevel.py* (update the location to match your directory structure - the location should show up in the error), this fixes the problem. Alternatively, replace the entire file with the file provided in this repo (you may need to restart the kernel after this):

```
mydir="/projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/"
cp toplevel.py $mydir
```

  This is the error I got when trying to run `demo_training_functions.py`:

```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
File ~/CNN_Spectrogram_Algorithm/Demo-Training/demo_training_functions.py:4
      2 sys.path.insert(1, '../')
      3 import matplotlib.pyplot as plt
----> 4 from fastai_v1.imports import *
      5 from fastai_v1.transforms import *
      6 from fastai_v1.conv_learner import *

File ~/CNN_Spectrogram_Algorithm/Demo-Training/../fastai_v1/imports.py:5
      3 import PIL, os, numpy as np, math, collections, threading, json, random, scipy, cv2
      4     # Don't import bcolz - it's not maintained anymore
----> 5 import bcolz
      6 import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
      7 import seaborn as sns, matplotlib

File /projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/__init__.py:81
     76 from bcolz.carray_ext import (
     77     carray, blosc_version, blosc_compressor_list,
     78     _blosc_set_nthreads as blosc_set_nthreads,
     79     _blosc_init, _blosc_destroy)
     80 from bcolz.ctable import ctable
---> 81 from bcolz.toplevel import (
     82     print_versions, detect_number_of_cores, set_nthreads,
     83     open, fromiter, arange, zeros, ones, fill,
     84     iterblocks, cparams, walk)
     85 from bcolz.chunked_eval import eval
     86 from bcolz.defaults import defaults, defaults_ctx

File /projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/toplevel.py:214
    210     obj.flush()
    211     return obj
--> 214 def fill(shape, dflt=None, dtype=np.float, **kwargs):
    215     """fill(shape, dtype=float, dflt=None, **kwargs)
    216 
    217     Return a new carray or ctable object of given shape and type, filled with
   (...)
    242 
    243     """
    245     def fill_helper(obj, dtype=None, length=None):

File /projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/numpy/__init__.py:305, in __getattr__(attr)
    300     warnings.warn(
    301         f"In the future `np.{attr}` will be defined as the "
    302         "corresponding NumPy scalar.", FutureWarning, stacklevel=2)
    304 if attr in __former_attrs__:
--> 305     raise AttributeError(__former_attrs__[attr])
    307 # Importing Tester requires importing all of UnitTest which is not a
    308 # cheap import Since it is mainly used in test suits, we lazy import it
    309 # here to save on the order of 10 ms of import time for most users
    310 #
    311 # The previous way Tester was imported also had a side effect of adding
    312 # the full `numpy.testing` namespace
    313 if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

</details>
