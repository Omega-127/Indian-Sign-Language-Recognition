import re
import tempfile
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import mediapipe as mp
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from gtts import gTTS

checkpoint_path = 'isl_bilstm_checkpoint.pt'
top_k = 3
translation_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

langauges = {
    "Hindi": ("hin_Deva", "hi"),
    "Marathi": ("mar_Deva", "mr"),
    "Tamil": ("tam_Taml", "ta")
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

@st.cache_resource(show_spinner="Loading sign recognition model...")
def load_sign_model():
    device = torch.device("cuda" if     torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    idx_to_label = checkpoint["idx_to_label"]
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    model = BiLSTMclassifier(
        input_size  = checkpoint.get("input_size", 1692),
        hidden_size = checkpoint.get("hidden_size", 128),
        num_layers  = checkpoint.get("num_layers", 1),
        num_classes = checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_label, device, checkpoint.get("test_accuracy", 0)

@st.cache_resource(show_spinner="Loading translation model (first run downloads ~800MB)...")
def load_translation_model(_device):
    tokenizer = AutoTokenizer.from_pretrained(translation_model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        translation_model_name, trust_remote_code=True
    ).to(_device)
    model.eval()
    ip = IndicProcessor(inference=True)
    return tokenizer, model, ip

# @st.cache_resource(show_spinner="Loading landmarks detectors...")
def load_landmakrers():
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
    return Handlandmarker, PoseLandmarker, FaceLandmarker


def process_video(video_path, Handlandmarker, PoseLandmarker, FaceLandmarker):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print(f"DEBUG: video fps = {fps}")
    frame_count_meta = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    sequence = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_idx / fps) * 1000)

        hand_result = Handlandmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = PoseLandmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = FaceLandmarker.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand = split_hands(hand_result)
        pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None

        sequence.append(extract_landmarks(pose_landmarks, face_landmarks, left_hand, right_hand))
        frame_idx += 1

    cap.release()
    print(f"DEBUG: processed {frame_idx} frames")
    end_ts = int((frame_idx / fps) * 1000) + 1000
    if len(sequence) == 0:
        return None
    return sequence

def translate_text(text, tgt_lang_code, translation_model, tokenizer, ip, device):
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=tgt_lang_code)
    inputs = tokenizer(batch, padding=True, max_length=256, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256,
            num_beams=5, num_return_sequences=1
        )

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang=tgt_lang_code)[0]

def synthesize_speech(text, gtts_lang_code):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_path = tmp.name
    tmp.close()
    tts = gTTS(text=text, lang=gtts_lang_code)
    tts.save(temp_path)
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
    Path(temp_path).unlink(missing_ok=True)
    return audio_bytes
    

#streamlit frontend

st.set_page_config(page_title="ISL sign translator", page_icon="👌", layout="centered")
st.title("ISL Sign Language Translator")
st.caption("Real time Indian sign langauge predictor + Regional Language Translation")
with st.expander("About this demo", expanded=False):
    st.markdown("""
    This demo recognizes one of **50 ISL words** from a short video clip,
    then translates and speaks the result in Hindi, Marathi, or Tamil.

    **Model accuracy:** 34.6% signer-independent test accuracy (18× better
    than random chance on 50 classes). Performance is strongest on
    well-sampled classes like *Car*, *Death*, *Thank you*, and *train ticket*
    — see the project README for the full confusion matrix and per-class
    breakdown.

    **Recording tips:**
    - Keep your full arm span in frame, not just your upper body at rest
    - Perform one complete sign, 1.5–4 seconds
    - Good, even lighting on your hands and face
    """)

model, idx_to_label, device, test_accuracy = load_sign_model()
# hand_landmarker, pose_landmarker, face_landmarker = load_landmakrers()
# st.success(f"Sign recogition model loaded : {len(idx_to_label)} classes, {test_accuracy:.1%} test accuracy.")
st.divider()

# if "running_timestamp" not in st.session_state:
st.session_state.running_timestamp = 0
st.write(f"DEBUG: running_timestamp = {st.session_state.running_timestamp}")
uploaded_file = st.file_uploader(
    "Upload a short video of one ISL sign (MP4, MOV, AVI)",
    type=["mp4", "mov", "avi"]
)
target_language = st.radio("Translate to: ", list(langauges.keys()), horizontal=True)
if uploaded_file is not None:
    st.video(uploaded_file)
    st.write(f"DEBUG: running_timestamp = {st.session_state.running_timestamp}")
    if st.button("Recognize & translate", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        with st.spinner("Extracting landmarks..."):
            hand_landmarker, pose_landmarker, face_landmarker = load_landmakrers()
            frames = process_video(video_path, hand_landmarker, pose_landmarker, face_landmarker)
            hand_landmarker.close()
            pose_landmarker.close()
            face_landmarker.close()
            # frames, new_ts = process_video(video_path, hand_landmarker, pose_landmarker, face_landmarker, 
            #                             start_ts=st.session_state.running_timestamp)
            # st.session_state.running_timestamp = new_ts
        Path(video_path).unlink(missing_ok=True)
        if frames is None:
            st.error("Could not process video - try a different file")
        else:
            trimmed = trim(frames)
            if len(trimmed) < 4:
                st.warning("" \
                f"Only {len(trimmed)} frames with detected hands - too few active frames. try a clip with clearer hand visibility")
            else:
                with st.spinner("Classifying sign..."):
                    results = classify_clip(model, trimmed, device, idx_to_label, top_k)
                st.subheader("Top predictions")
                for label, prob in results:
                    st.write(f"**{label}** - {prob:.1%}")
                    st.progress(prob)

                top_label = results[0][0]
                gloss = clean_gloss(top_label)
                tgt_lang_code, gtts_code = langauges[target_language]
                with st.spinner(f"Loading translation model and translating to {target_language}..."):
                    tokenizer, translation_model, ip = load_translation_model(device)
                    translated = translate_text(gloss, tgt_lang_code, translation_model, tokenizer, ip, device)
                    st.subheader(f"Translation {target_language}")
                    st.markdown(f"### {translated}")
                    with st.spinner("Generating speech..."):
                        try:
                            audio_bytes = synthesize_speech(translated, gtts_code)
                            st.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Speech synthesis failed check innternet connection {e}")
else:
    st.info("Upload a video to get started")