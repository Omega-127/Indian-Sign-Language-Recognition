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

langauges = {
    "1": ("hin_Deva", "hi", "Hindi"),
    "2": ("mar_deva", "mr", "Marathi"),
    "3": ("tam_taml", "ta", "Tamil")
}

base_options = mp.tasks.BaseOptions
hand_landmarker = mp.tasks.vision.HandLandmarker
hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
face_landmarker = mp.tasks.vision.FaceLandmarker
face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks

pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
face_connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

class BiLSTMclassifier(nn.Module):
    def __init__(self, input_size=1692, hidden_size=128, num_layers=1, num_classes=50, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, length):
        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, _) = self.lstm(packed)

        forward_h = hidden[-2]
        backward_h = hidden[-1]
        combined = torch.cat([forward_h, backward_h], dim=1)

        out = self.dropout(combined)
        out = self.classifier(out)
        return out
    
def extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand):
    def flatten(landmark_list, count, with_visiblity=False):
        if not landmark_list:
            return np.zeros(count * (4 if with_visiblity else 3))
        if with_visiblity:
            return np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmark_list]).flatten()
        return np.array(
                [[lm.x, lm.y, lm.z] for lm in landmark_list]).flatten()
    
    pose = flatten(pose_landmarks, 33, with_visiblity=True)
    face = flatten(face_landmarks, 478)
    left = flatten(left_hand, 21)
    right = flatten(right_hand, 21)

    return np.concatenate([pose, face, left, right])

def split_hands(hand_result):
    left_hand, right_hand = None, None
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name
        if label == 'Left':
            left_hand = landmarks
        elif label == 'Right':
            right_hand = landmarks
    return left_hand, right_hand

def classify_clip(model, frames, device, idx_to_label, top_k=3):
    sequence = np.array(frames, dtype=np.float32)
    tensor = torch.tensor(sequence).unsqueeze(0).to(device)
    length = torch.tensor([len(frames)])   
    with torch.no_grad():
        output = model(tensor, length)
        probs = torch.softmax(output, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = idx_to_label.get(idx, "?")
        results.append((label, prob))
    return results

def trim(frames):
    def has_hands(frame):
        left_hand_data = frame[1566:1629]
        right_hand_data = frame[1629:1692]
        return np.any(left_hand_data != 0) or np.any(right_hand_data != 0)
    active_indices = [i for i, f in enumerate(frames) if has_hands(f)]
    if not active_indices:
        return frames
    start = active_indices[0]
    end = active_indices[-1] + 1
    return frames[start:end]

def clean_gloss(label):
    return re.sub(r"^\d+\.\s*", "", label)

