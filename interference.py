import time
import json 
from collections import deque, Counter
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import mediapipe as mp

#configs
checkpoint_path = 'isl_bilstm_checkpoint.pt'
window_size = 60
inferenc_interval = 5
smoothing_history = 5
confidence_threshold = 0.3

#shortcuts
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
        self.lstm == nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
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

def run_interference(model, window, device):
    sequence = np.array(window, dtype=np.float32)
    tensor = torch.tensor(sequence).unsqueeze(0).to(device)
    length = torch.tensor([len(window)])

    with torch.no_grad():
        output = model(tensor, length)
        probs = torch.softmax(output, dim = 1)
        confidence, pred_idx = torch.max(probs, dim=1)
    return pred_idx.item(), confidence.item()

def main():
    


if __name__ == "__main__":
    main()

