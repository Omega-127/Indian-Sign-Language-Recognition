import cv2
import numpy as np
import mediapipe as mp
import time

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

#connection constants
pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
face_connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS


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

def main():
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
    recording = False
    sequence = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        print("Frame:", ret, frame.shape if ret else None)
        if not ret:
            break

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
            sequence.append(extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand))
            cv2.putText(frame, "RECORDING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        cv2.imshow("ISL LANDMARK EXTRACTION - STAGE 1 (TASKS API)", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('r'):
            recording = not recording
            if not recording and sequence:
                np.save("C:\\Users\\Sarvesh Haldikar\\projects\\landmarks.npy", np.array(sequence))
                print(f"saved {len(sequence)} frames to landmarks.npy")
                sequence = []
        elif key == ord('q'):
            break

    Handlandmarker.close()
    PoseLandmarker.close()
    FaceLandmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()