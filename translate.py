import re
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import mediapipe as mp


#configs
checkpoint_pat = 'isl_bilstm_checkpoint.pt'
top_k = 3
translation_model = "ai4bharat/indictrans2-en-indic-dist-200M"