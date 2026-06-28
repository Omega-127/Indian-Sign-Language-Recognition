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

def extract_landmarks(pose_landmarks, face_landmarks,left_hand, right_hand):
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


def process_video(video_path, hand_lm, pose_lm, face_lm, start_ts=0):
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


        timestamp_ms = start_ts + int((frame_idx/fps)*1000)

        hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
        face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand = split_hands(hand_result)
        pose_landmarks = pose_result.pose_landmarks[0] \
            if pose_result.pose_landmarks else None
        face_landmarks = face_result.face_landmarks[0] \
            if face_result.face_landmarks else None
        
        sequence.append(extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand))
        frame_idx += 1

    cap.release()

    if len(sequence) == 0:
        return None, 0

    return np.array(sequence), int((frame_idx / fps) * 1000)


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
        base_options=BaseOptions(model_asset_path="c:\\Users\\Sarvesh Haldikar\\projects\\hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands = 2
    ))

    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="c:\\Users\\Sarvesh Haldikar\\projects\\pose_landmarker_lite.task"),
        running_mode=VisionRunningMode.VIDEO
    ))

    face_lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="c:\\Users\\Sarvesh Haldikar\\projects\\face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO
    ))

    
    index_rows = []
    skipped = []
    processed = 0
    running_timestamp = 0

    for i, row in enumerate(df_unique.itertuples(), 1):
        video_path = videos_dir/row.video_path
        label = row.label
        class_idx = label_to_idx[label]
        split = row.split

        npy_path = output_dir/Path(row.video_path).with_suffix(".npy")


        if npy_path.exists():
            print(f"[{i:03d}/{len(df_unique)}]  Already done: {row.video_path}")
            index_rows.append({
                "video_path": row.video_path,
                "npy_path": str(npy_path),
                "label": class_idx,
                "split": split
            })
            continue


        if not video_path.exists():
            print(f"[{i:03d}/{len(df_unique)}]  File missing: {video_path}")
            skipped.append(str(video_path))
            continue

        print(f"[{i:03d}/{len(df_unique)}] Processing: {row.video_path}", end="", flush=True)

        npy_path.parent.mkdir(parents=True, exist_ok=True)

        # extract landmarks from this video
        result = process_video(video_path, hand_lm, pose_lm, face_lm, running_timestamp)

        if result is None:
            print(f"  Failed")
            skipped.append(str(video_path))
            running_timestamp += 5000
            continue

        sequence, frames_used = result
        running_timestamp += frames_used + 1000

        

        #save the landamrk sequence
        np.save(str(npy_path), sequence)
        processed += 1

        print(f"{sequence.shape[0]} frames, shape {sequence.shape}")

        index_rows.append({
            "video_path": row.video_path,
            "npy_path": str(npy_path),
            "label": label,
            "class_index": class_idx,
            "split": split
        })

    hand_lm.close()
    pose_lm.close()
    face_lm.close()

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(index_csv, index=False)

    print()
    print("=" * 60)
    print("Done.")
    print(f"Newly extracted: {processed} videos")
    print(f"Skipped/failed: {len(skipped)} videos")
    print(f"Index saved to: {index_csv}")
    print(f"Label map: label_map.csv")
    print("=" * 60)

    if skipped:
        print(f"\nFailed videos: ({len(skipped)}):")
        for s in skipped:
            print(f" - {s}")


if __name__ == "__main__":
    main()


