import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import mediapipe as mp

#config
videos_dir = Path("videos")
output_dir = Path("extracted_landmarks")
metadata_csv = "include50_metadata.csv"
index_csv = "landmarks_index.csv"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def extract_landmarks(pose_landmarks, face_landmarks, with_visibilty = False):
    def flatten(landmark_list, count, with_visibilty=False):
        if not landmark_list:
            return np.zeros(count*(4 if with_visibilty else 3))
        if with_visibilty:
            return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmark_list]).flatten()
        return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list]).flatten()
    
    pose = flatten(pose_landmarks, 33, with_visibilty=True)
    face = flatten(face_landmarks, 478)
    left = flatten(left_hand, 21)
    right = flatten(right_hand, 21)
    return np.concatenate([pose, face, left, right])

def split_hands(hand_result):
    left_hand, right_hand = None, None
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name
        if label == "Left":
            left_hand = landmarks
        if label == "Right":
            right_hand = landmarks
    return left_hand, right_hand


def process_video(video_path, hand_lm, pose_lm, face_lm):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30    #fallback if video could not report fps

    sequence = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


        timestamp_ms = int({frame_idx/fps}*1000)

        hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
        face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand = split_hands(hand_result)
        pose_landmarks = pose_result.pose_landmarks[0] \
            if pose_result.pose_landmarks else None
        face_landmarks = face_result.face_landmarks[0] \
            if face_result.face_landmarks else None
        
        sequence.append(extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand))
        frame += 1

    cap.release()

    if len(sequence) == 0:
        return None
    
    return np.array(sequence) #shape(num_frames, 1692)


def main():
    if not Path(metadata_csv).exists():
        print(f"Error: {metadata_csv} not found. Run metadata.py first")
        sys.exit(1)

    df = pd.read_csv(metadata_csv)

    unique_labels = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Classes: {len(unique_labels)}")

    #Save the label mapping so training and inference use the same indices

    label_mp_df = pd.DataFrame(list(label_to_idx.items()), columns=["label", "class_index"])
    label_mp_df.to_csv("label_map.csv", index=False)
    print(f"Saved label mapping to label_map.csv")
    
    df_unique = df.drop_duplicates(subset="video_path", keep="first")
    print(f"Videos to process: {len(df_unique)}")
    print()

    output_dir.mkdir(exist_ok=True)

    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands = 2
    ))

    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="psoe_landmarker_lite.task"),
        running_mode=VisionRunningMode.VIDEO
    ))

    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmaker.task"),
        running_mode=VisionRunningMode.VIDEO
    ))

    
    index_rows = []
    skiepped = []
    processed = 0
    
