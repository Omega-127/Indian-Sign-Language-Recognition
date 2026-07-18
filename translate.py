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
checkpoint_path = 'isl_bilstm_checkpoint.pt'
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

def translate_text(text, tgt_lang_code, translation_model, tokenizer, ip, device):
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lng=tgt_lang_code)
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs, use_catch = True, min_length=0,
            max_length=256, num_beams=5, num_return_sequence=1
        )

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    translated = ip.postprocess_batch(decoded, lang=tgt_lang_code)[0]
    return translated

def speak_text(text, gtts_lang_code):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name
    tts = gTTS(text=text, lang=gtts_lang_code)
    tts.save(temp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()

    Path(temp_path).unlink(missing_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    idx_to_label = checkpoint['idx_to_label']

    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    model = BiLSTMclassifier(
        input_size  = checkpoint.get("input_size", 1692),
        hidden_size = checkpoint.get("hidden_size", 128),
        num_layers  = checkpoint.get("num_layers", 1),
        num_classes = checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded sign recognition model: {checkpoint['num_classes']} classes, "
        f"test accuracy {checkpoint.get('test_accuracy', 0):.1%}")
    
    print(f"Loading IndicTrans2 translation model (first run dowloads 800mb)...")
    tokenizer = AutoTokenizer.from_pretrained(translation_model, trust_remote_code=True)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        translation_model, trust_remote_code=True
    ).to(device)

    Handlandmarker = hand_landmarker.create_from_options(hand_landmarker_options(
        base_options = base_options(model_asset_path="hand_landmarker.task"),
        running_mode = vision_running_mode.VIDEO,
        num_hands = 2,
        min_hand_detection_confidence = 0.3,
        min_hand_presence_confidence = 0.3,
        min_tracking_confidence = 0.3
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
    last_translation = ""
    current_lang = "1"

    print("\nContols:")
    print("r: start/stop recording")
    print("1/2/3: switch language (Hindi, Marathi, Tamil)")
    print("q: quit\n")
    print(f"Current language: {langauges[current_lang][2]}\n")

    while cap.isOpened():
        ret, frame = cap.read()
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

        hands_status = f"L:{'OK' if left_hand else 'X'}  R:{'OK' if right_hand else 'X'}"
        cv2.putText(frame, hands_status, (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Lang: {langauges[current_lang][2]}", (15, frame.shape[0] - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if recording:
            if left_hand is None and right_hand is None:
                hand_missing_count += 1
            landmarks = extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand)
            recorded_frames.append(landmarks)
            cv2.putText(frame, f"Recording ({len(recorded_frames)} frames)", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            

        if last_results and not recording:
            y = frame.shape[0] - 20 - (len(last_results) - 1) * 30
            for label, prob in last_results:
                text = f"{label} ({prob:.0%})"
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30

        cv2.imshow("ISL sign translation - Stage 4", frame)
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
                np.save("debug_last_recording.npy", np.array(recorded_frames))
                print("Saved debug_last_recording.npy for inspection")
                trimmed_frames = trim(recorded_frames)
                print(f"Trimmed to {len(trimmed_frames)} active frames from {len(recorded_frames)}")
                if len(trimmed_frames) >= 4:
                    last_results = classify_clip(model, trimmed_frames, device, idx_to_label, top_k)
                    print("Top predictions: ")
                    for label, prob in last_results:
                            print(f"{label:<25} {prob:.1%}")
                        
                    top_label = last_results[0][0]
                    gloss = clean_gloss(top_label)
                    tgt_lang_code, gtts_code, lang_name = langauges[current_lang]
                    print(f"Translating {gloss} to {lang_name}")
                    translated = translate_text(gloss, tgt_lang_code, translation_model, tokenizer, ip, device)
                    last_translation = translated
                    print(f"-> {translated}")
                    print("speaking...")
                    try:
                        speak_text(translated, gtts_code)
                    except Exception as e:
                        print(f"speech failed check internet conection: {e}")

                else:
                    print("Too few active frames...")
                    last_results = []

        elif key in (ord('1'), ord('2'), ord('3')):
            current_lang = chr(key)
            print(f"language switched to : {langauges[current_lang][2]}")
        elif key == ord('q'):
            break

    Handlandmarker.close()
    PoseLandmarker.close()
    FaceLandmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
        main()
