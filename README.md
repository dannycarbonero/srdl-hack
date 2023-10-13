# Project: Automated spike-ripple detection using deep learning

This is as an example on how teams can structure their project repositories. Thanks to Lindsey Heagey and Joachim Meyer for the template!

This is forked from the code corresponding to the paper with minor updates relating to package versions.
https://github.com/eschlaf2/CNN_Spectrogram_Algorithm

## Troubleshooting

You might get an error when you try to import `bcolz` because it assumes an older version of NumPy. 

<details>
  <summary>bcolz/np.float error</summary>
  
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

If you replace all instances of `np.float` with `float` in */projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/toplevel.py*, this fixes the problem. Alternatively, replace the entire file with the file provided in this repo (you may need to restart the kernel after this):

```
mydir="/projectnb/ecog/eds2/.conda/envs/cnn-specgram/lib/python3.8/site-packages/bcolz/"
cp toplevel.py $mydir
```

</details>


## Files

* `.gitignore`
<br> Globally ignored files by `git` for the project.
* `environment.yml`
<br> `conda` environment description needed to run this project.
* `README.md`
<br> Description of the project (see suggested headings below)

## Folders

### `contributors`
Each team member has it's own folder under contributors, where they can work on their contribution. Having a dedicated folder for each person helps to prevent conflicts when merging with the main branch.

### `notebooks`
Notebooks that are considered delivered results for the project should go in here.

### `scripts`
Helper utilities that are shared with the team

# Recommended content for your README.md file:

## Project Summary

### Project Title

Brief title describing the proposed work.

### Collaborators on this project

List all participants on the project. Choose one team member to act as project lead, and identify one hackweek organizer as the data science lead.

### The problem

What problem are you going to explore? Provide a few sentences. If this is a technical exploration of software or data science methods, explain why this work is important in a broader context.

### Application Example

List one specific application of this work.

### Sample data

If you already have some data to explore, briefly describe it here (size, format, how to access).

### Specific Questions

List the specific tasks you want to accomplish or research questions you want to answer.

### Existing methods

How would you or others traditionally try to address this problem?

### Proposed methods/tools

Building from what you learn at this hackweek, what new approaches would you like to try to implement?

### Background reading

Nadalin, Jessica K., Uri T. Eden, Xue Han, R. Mark Richardson, Catherine J. Chu, and Mark A. Kramer. 2021. “Application of a Convolutional Neural Network for Fully-Automated Detection of Spike Ripples in the Scalp Electroencephalogram.” Journal of Neuroscience Methods 360 (August): 109239. https://doi.org/10.1016/j.jneumeth.2021.109239.


