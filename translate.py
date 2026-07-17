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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import pygame
from gtts import gTTS

#configs
checkpoint_pat = 'isl_bilstm_checkpoint.pt'
top_k = 3
translation_model = "ai4bharat/indictrans2-en-indic-dist-200M"