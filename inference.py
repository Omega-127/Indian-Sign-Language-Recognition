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
top_k = 3

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    idx_to_label = checkpoint['idx_to_label']

    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    model = BiLSTMclassifier(
        input_size = checkpoint.get("input_size", 1692),
        hidden_size = checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 1),
        num_classes=checkpoint["num_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: {checkpoint['num_classes']} classes, test accuracy {checkpoint.get('test_accuracy', 0):.1%}")

    Handlandmarker = hand_landmarker.create_from_options(hand_landmarker_options(
        base_options = base_options(model_asset_path="hand_landmarker.task"),
        running_mode = vision_running_mode.VIDEO,
        num_hands = 2
    ))

    PoseLandmarker = pose_landmarker.create_from_options(pose_landmarker_options(
        base_options = base_options(model_asset_path="pose_landmarker_lite.task"),
        running_mode = vision_running_mode.VIDEO
    ))

    FaceLandmarker = face_landmarker.create_from_options(face_landmarker_options(
        base_options = base_options(model_asset_path="face_landmarker.task"),
        running_mode = vision_running_mode.VIDEO
    ))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start_time = time.time()
    
    recording = False
    hand_missing_count = 0
    recorded_frames = []
    last_results = []

    print("Press 'r' to start/stop recording one sign. Press 'q' to quit.\n")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time)*1000)

        hand_result = Handlandmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = PoseLandmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = FaceLandmarker.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand = split_hands(hand_result)
        pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None

        if face_landmarks:
            draw_landmarks(frame, face_landmarks, face_connections)
        if pose_landmarks:
            draw_landmarks(frame, pose_landmarks, pose_connections)
        if left_hand:
            draw_landmarks(frame, left_hand, hand_connections)
        if right_hand:
            draw_landmarks(frame, right_hand, hand_connections)

        if recording:
            if left_hand is None and right_hand is None:
                hand_missing_count += 1
            landmarks = extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand)
            recorded_frames.append(landmarks)
            cv2.putText(frame, f"Recording ({len(recorded_frames)} frames)", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            hands_status = f"L:{'OK' if left_hand else 'X'}  R:{'OK' if right_hand else 'X'}"
            cv2.putText(frame, hands_status, (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if last_results and not recording:
            y = frame.shape[0] - 20 - (len(last_results) - 1) * 30
            for label, prob in last_results:
                text = f"{label} ({prob:.0%})"
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30

        cv2.imshow("ISL sign recoginition - Stage 3", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if recording:
                recorded_frames = []
                hand_missing_count = 0
                last_results = []
                print("Recording started...")
            else:
                print(f"Recording stopped. {len(recorded_frames)} frames captured.")
                print(f"Frames with NO hands detected: {hand_missing_count}/{len(recorded_frames)}")
                if len(recorded_frames) >= 4:
                    last_results = classify_clip(model, recorded_frames, device, idx_to_label, top_k)
                    print("DEBUG last_results:", last_results)
                    print("Top predictions: ")
                    for label, prob in last_results:
                            print(f"{label:<25} {prob:.1%}")
                else:
                    print("Too few frames captured -- try recording a longer clip.")
                    last_results = []
        elif key == ord('q'):
            break

            
    Handlandmarker.close()
    PoseLandmarker.close()
    FaceLandmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

