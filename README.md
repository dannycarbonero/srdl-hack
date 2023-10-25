# Project: Automated spike-ripple detection using deep learning

The Kramer-Eden-Chu lab hackathon to use deep learning to create an automated spike-ripple detector. This builds on the project in [Nadalin et al., 2021](https://doi.org/10.1016/j.jneumeth.2021.109239). Exact aims will depend on what we decide to do during the hackathon, but some examples include

* Use a model pre-trained on a different set of images (e.g., satellite imagery instead of natural images)
* Account for expected artifacts (e.g., broadband increases in power)
* Use an LSTM-based RNN
* Augment the training dataset

### Possible resources

* Nadalin, Jessica K., Uri T. Eden, Xue Han, R. Mark Richardson, Catherine J. Chu, and Mark A. Kramer. 2021. “**Application of a Convolutional Neural Network for Fully-Automated Detection of Spike Ripples in the Scalp Electroencephalogram.**” Journal of Neuroscience Methods 360 (August): 109239. https://doi.org/10.1016/j.jneumeth.2021.109239. [[Code](https://github.com/Eden-Kramer-Lab/CNN_Spectrogram_Algorithm)]

* Hagen, Espen, Anna R. Chambers, Gaute T. Einevoll, Klas H. Pettersen, Rune Enger, and Alexander J. Stasik. 2021. “**RippleNet: A Recurrent Neural Network for Sharp Wave Ripple (SPW-R) Detection.**” Neuroinformatics 19 (3): 493–514. https://doi.org/10.1007/s12021-020-09496-2. [[Code](https://github.com/CINPLA/RippleNet)]

* Sarmashghi, Mehrad, Shantanu P. Jadhav, and Uri T. Eden. 2022. “**Integrating Statistical and Machine Learning Approaches for Neural Classification.**” IEEE Access 10: 119106–18. https://doi.org/10.1109/ACCESS.2022.3221436. 

* [MATLAB Deep Learning Course](https://matlabacademy.mathworks.com/details/mldl) ([Course quick-reference](https://matlabacademy.mathworks.com/artifacts/quick-reference.html?course=mldl&language=en&release=R2023a))
    * [Deep learning tips and tricks](https://www.mathworks.com/help/deeplearning/ug/deep-learning-tips-and-tricks.html)
* Free [deep learning ebook](https://www.deeplearningbook.org/)
*	[3b1b explanation of neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
*	[Fastai](https://www.fast.ai/) (course, tutorials, fastai for PyTorch)
    *	Coincidentally, [this interview](https://podcasters.spotify.com/pod/show/nobila-nadhira/episodes/Episode-391-Jeremy-Howard-on-Deep-Learning-and-fast-ai-ek6fos) with Jeremy Howard of fastai just came out - he's very interested in making deep learning and AI accessible
*	[Google Machine Learning Education](https://developers.google.com/machine-learning)


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

* Nadalin, Jessica K., Uri T. Eden, Xue Han, R. Mark Richardson, Catherine J. Chu, and Mark A. Kramer. 2021. “Application of a Convolutional Neural Network for Fully-Automated Detection of Spike Ripples in the Scalp Electroencephalogram.” Journal of Neuroscience Methods 360 (August): 109239. https://doi.org/10.1016/j.jneumeth.2021.109239. [[Code](https://github.com/Eden-Kramer-Lab/CNN_Spectrogram_Algorithm)]


