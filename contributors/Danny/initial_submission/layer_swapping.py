import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from contributors.Danny.initial_submission.utilities import load_RippleNet, binarize_RippleNet

RippleNet = load_RippleNet('code')
RippleNet_bin = binarize_RippleNet(RippleNet)