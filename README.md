# Real-Time Sign Language Translator for Indian Regional Languages

A multi-stage machine learning project to detect and translate Indian Sign Language (ISL) into regional Indian languages (Hindi, Marathi, Tamil, etc.) with real-time inference and text-to-speech output.

## Why This Project?

Most sign language translation projects focus on American Sign Language (ASL). ISL remains severely under-resourced — no real-time translators exist, and the gap between research and deployed tools is massive. This project aims to change that by building an **end-to-end, production-ready pipeline** that bridges computer vision, sequence modeling, NLP, and speech synthesis.

---

## Project Status

### ✅ Stage 1 — Complete
**Real-time Landmark Extraction Baseline**
- MediaPipe Holistic (Hand + Pose + Face) landmark detection from live webcam
- Extracts and saves landmark sequences as NumPy arrays
- Array shape: `(frames, 1692)` — fixed-length per-frame representation

**Files:**
- `main.py` — main script

**Output:**
- `landmarks.npy` — saved landmark sequences from recorded signing

### ✅ Stage 2 — Complete
**Bi-LSTM Word Recognition on INCLUDE-50**
- Downloaded and extracted 943 ISL sign videos (675 train / 77 val / 191 test) from 50 word classes across 15 categories
- Built Bi-LSTM sequence classifier with data augmentation (Gaussian noise, time-scaling, spatial-scaling)
- Trained with early stopping, achieving **36.6% signer-independent test accuracy** (18× better than random chance on 50 classes)

**Key Technical Findings:**
- **Signer-Independent Generalization:** Validation accuracy reached ~55% while test accuracy settled at 36.6%. This gap reveals a crucial insight: the model learned to recognize signs but also partly memorized individual signer characteristics. The test set contains signers not present in training — a known research challenge in sign language recognition.
- **Data Augmentation Impact:** Adding proper shuffling + augmentation increased validation accuracy from 44% to 55%, confirming the importance of data augmentation on small datasets (675 examples).
- **Architecture Optimization:** Smaller model (128 hidden units, 1 layer) with dropout=0.5 outperformed larger architectures, indicating that constraining model capacity was more effective than increasing it for this dataset size.

**Model Performance by Class:**
- **Perfect (100% recall):** "16. train ticket", "2. Death", "28. Window", "86. Time", "84. Teacher"
- **Strong (80%+):** "1. Dog" (83%), "1. loud" (92%), "11. Car" (93%), "23. Court" (100%), "28. Store or Shop" (100%)
- **Weak (<30% recall):** "19. House" (0%), "34. Pen" (27%), "53. Fan" (18%), "47. Red" (12%), "44. it" (9%)
- Classes with poorest performance had the fewest training examples (5-11 clips)

**Files:**
- `extract_keypoints.py` — extract landmarks from all 943 INCLUDE-50 videos
- `ISL.ipynb` — Colab notebook for Bi-LSTM training (10 cells)
- `extracted_landmarks/` — 943 .npy files with landmark sequences per video
- `landmarks_index.csv` — metadata mapping videos to labels, class indices, and train/val/test splits
- `label_map.csv` — 50-class label-to-index mapping for model inference
- `isl_bilstm_checkpoint.pt` — trained model weights + full metadata for Stage 3 inference
- `training_curves_final.png` — loss and accuracy plots across 80 epochs
- `confusion_matrix_final.png` — 50×50 confusion matrix showing per-class predictions

**Training Curves:**
<img width="2100" height="750" alt="training_curves (1)" src="https://github.com/user-attachments/assets/a45b2975-f891-4988-886a-14f43e3f1198" />

*Loss and accuracy across 80 epochs. Train loss decreases smoothly while val accuracy plateaus around 55%, reflecting the signer-independent generalization challenge.*

**Confusion Matrix (Test Set):**
<img width="3000" height="2700" alt="confusion_matrix (1)" src="https://github.com/user-attachments/assets/43db0b2f-e874-4c70-84ae-cb610ef7ae6e" />

*50×50 matrix showing predicted vs. true class for all 191 test samples. Diagonal blocks (dark blue) indicate correct predictions. Classes like "16. train ticket", "2. Death", "86. Time" achieve near-perfect recall. Classes with few training examples (rows like "19. House", "34. Pen") have zero correct predictions, indicating data scarcity is the limiting factor.*

**Model Architecture:**
```
Input: (batch_size, variable_frames, 1692 landmarks)
  ↓
Bi-LSTM: 2 layers → 128 hidden units, bidirectional
  ↓
Last hidden state concatenation: 128*2 = 256 dims
  ↓
Dropout (p=0.5) → Linear(256 → 50)
  ↓
Output: 50-class logits (softmax at inference)

Total parameters: 1.8M
```

**Training Details:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Batch size: 32
- Max epochs: 80 with early stopping (patience=12)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=4)
- Data augmentation: applied only to training, not val/test

### ✅ Stage 3 — Complete
**Real-Time Sign Recognition (Record-and-Classify)**
- Integrated the trained Bi-LSTM with Stage 1's webcam landmark pipeline
- Initial approach (continuous sliding-window prediction) was abandoned after testing — the model was trained on complete, isolated sign clips, so classifying arbitrary continuous-video fragments produced near-random results
- Rebuilt around a **record-then-classify** pattern matching training conditions: press `r` to start recording one sign, press `r` again to stop, and the complete captured clip is classified in a single pass — the same way each INCLUDE-50 training video was handled
- Added automatic dead-frame trimming (cuts leading/trailing frames with no hand detected) since manual start/stop recording captures idle time the model never saw in training
- Displays top-3 predictions with confidence, rather than a single guess — often useful since correct answers frequently appear in position 2–3, especially for visually similar signs

**Debugging Journey (real findings, not just a working script):**
- **Confirmed the trained checkpoint itself is correct** by running it directly against saved INCLUDE-50 test clips — a "16. train ticket" test clip classified correctly at 69.8% confidence, matching Colab evaluation results
- **Diagnosed a hand-detection dropout issue**: live recordings showed 40-89% of frames with zero hands detected, versus clean detection on saved dataset videos. Root cause was recording technique — pressing record before/after the actual sign motion captured long stretches of idle frames, diluting the real signal
- **Diagnosed a pose-landmark scale mismatch**: live pose coordinates ranged roughly (-1.9, 2.98) versus training data's (-0.65, 1.3) — MediaPipe was extrapolating positions for body parts outside the camera's field of view. Repositioning further from the camera (to keep full arm extension in frame during signing) tightened this to (-1.07, 1.16), much closer to training conditions
- **Result:** with proper recording technique and camera framing, the pipeline produces plausible, varied top-3 predictions rather than degenerate/repeated guesses — confirming the extraction → trimming → inference chain works correctly end to end

**Honest Performance Expectation:** Given the model's 36.6% signer-independent test accuracy on clean, curated dataset clips, live recognition — using a signer (you) the model has never seen, on hardware/framing that can only approximate the original recording setup — should realistically succeed roughly 1 in 3 attempts, even under ideal conditions. This isn't a pipeline flaw; it's the same generalization gap documented in Stage 2, now visible firsthand.

**Files:**
- `inference.py` — real-time record-and-classify inference script

**Controls:**
- `r` — start/stop recording one sign
- `q` — quit

### ✅ Stage 4 — Complete
**Translation & Speech Output**
- Extended Stage 3's record-and-classify pipeline with translation and text-to-speech
- Recognized sign → cleaned gloss (strips the INCLUDE-50 class-number prefix, e.g. `"48. Hello"` → `"Hello"`) → translated via **IndicTrans2** (AI4Bharat's model) → spoken aloud via **gTTS**
- Supports three target languages with live switching: Hindi, Marathi, Tamil (keys `1`/`2`/`3`)
- Uses the distilled 200M IndicTrans2 model (`indictrans2-en-indic-dist-200M`) — lighter weight than the 1B variant, appropriate for single-word/short-phrase translation rather than long sentences

**Setup Issues Resolved (real findings):**
- **`transformers` version incompatibility:** `IndicTransToolkit` requires `transformers==4.53.2` specifically — newer versions (5.x) moved `PreTrainedTokenizerBase` to a different internal location, breaking the import. Pinning the exact version was required.
- **Virtual environment mismatch:** spent significant debugging time on an import error that persisted despite confirming the correct package version — root cause was two separate Python installations (a global install and a project `.venv`) with independent `site-packages`, so verifying a fix in one terminal didn't affect the other. Resolved by explicitly activating the `.venv` before installing.
- **Hugging Face authentication:** first model download attempt failed with a "gated repo" 401 error — resolved by authenticating via `huggingface-cli login`, despite the model being MIT-licensed and not actually restricted (misleading error message for unauthenticated requests).
- **Variable name collision:** reusing the same variable name for a model-name string and the loaded model object inside the same function caused an `UnboundLocalError`, due to Python treating the name as function-local as soon as any assignment to it exists anywhere in the function.

**Result:** Full pipeline working end-to-end — webcam recording → sign classification → gloss cleanup → regional-language translation → spoken audio output, with live language switching.

**Files:**
- `translate.py` — full pipeline: record → classify → translate → speak

**Controls:**
- `r` — start/stop recording one sign
- `1` / `2` / `3` — switch target language (Hindi / Marathi / Tamil)
- `q` — quit

---

## Technical Stack

### Stage 1 — Vision & Landmarks
- **MediaPipe 0.10.35** — Hand, Pose, and Face landmark detection via Tasks API
- **OpenCV** — webcam capture and real-time visualization
- **NumPy** — array operations and efficient storage

### Stage 2 — Sequence Modeling & Training
- **PyTorch** — Bi-LSTM implementation, training loop, early stopping
- **scikit-learn** — confusion matrix, classification metrics
- **Pandas** — dataset management and metadata tracking
- **Matplotlib + Seaborn** — training curves and confusion matrix visualization
- **Google Colab** — T4 GPU for efficient training (optional; CPU also works)

### Stage 3 — Real-Time Inference
- **PyTorch** — loading trained checkpoint, running live inference
- **MediaPipe Tasks API** — same three landmarkers as Stage 1, reused for live capture
- **OpenCV** — webcam capture, recording toggle, on-screen results display

### Stage 4 — Translation & Speech
- **IndicTrans2** (`indictrans2-en-indic-dist-200M`) — ISL gloss → Hindi/Marathi/Tamil translation
- **IndicTransToolkit** — pre/post-processing for IndicTrans2 (pinned `transformers==4.53.2`)
- **gTTS** — text-to-speech synthesis (requires internet)
- **pygame** — audio playback

### Stage 5 (Planned)
- **Streamlit or React** — frontend for deployment
- Demo video and final packaging

---

## Setup & Installation

### Prerequisites
- Python 3.9–3.12
- Windows/macOS/Linux
- Webcam (for Stage 1)
- ~2GB disk space (for model files)

### 1. Install Dependencies
```bash
pip install --upgrade mediapipe==0.10.35 opencv-python numpy
```

### 2. Download Model Files
Stage 1 requires three MediaPipe task bundles. Download and place them in the project folder:

```bash
# Hand landmark detection
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Pose landmark detection
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# Face landmark detection
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Or download via command line (Linux/macOS):
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 3. Verify Installation
```bash
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

---

## Usage: Stage 1

### Run the Real-Time Pipeline
```bash
python main.py
```

A window titled **"ISL LANDMARK EXTRACTION - STAGE 1 (TASKS API)"** will open showing your webcam feed with live landmark dots and skeleton overlays.

### Recording Landmark Sequences
1. **Press `r`** to start recording — you'll see "RECORDING" in red text
2. **Perform a sign** (hand gestures, body movement, facial expression)
3. **Press `r` again** to stop — this saves the sequence to `landmarks.npy` and prints:
   ```
   saved N frames to landmarks.npy
   ```
4. **Press `q`** to quit

### Verify Saved Data
```python
import numpy as np
data = np.load("landmarks.npy")
print(data.shape)  # Expected: (N, 1692) where N = frames recorded
print(data[0])     # First frame's 1692 landmark values
```

### Output Format
Each saved array has shape `(N, 1692)`:
- **Columns 0–131:** Pose landmarks (33 points × 4: x, y, z, visibility)
- **Columns 132–1565:** Face landmarks (478 points × 3: x, y, z)
- **Columns 1566–1628:** Left hand landmarks (21 points × 3: x, y, z)
- **Columns 1629–1692:** Right hand landmarks (21 points × 3: x, y, z)

All values are normalized to [0, 1] range (coordinates relative to frame size).

---

## Usage: Stage 3

### Setup
Place `isl_bilstm_checkpoint.pt` (trained model from Stage 2) in the same folder as `inference.py`, alongside the three `.task` model files.

### Run
```bash
python inference.py
```

### Recording and Classifying a Sign
1. **Click the video window** to give it keyboard focus (required for `r`/`q` to register)
2. **Press `r`** to start recording — timing matters: start right as you begin the sign motion
3. **Perform one complete sign**, then **press `r` again immediately** to stop
4. The top-3 predictions with confidence appear on screen and print to console
5. Press `r` to record another sign, `q` to quit

### Framing Tips (Important)
- Position yourself so your **full arm span at maximum extension** stays within the camera frame — not just your upper body at rest. Signs involve reaching motion that can extend beyond a typical video-call framing.
- Ensure good, even lighting on your hands and face
- Keep recordings tight — avoid pausing before/after the actual sign motion, since idle frames dilute the signal the model looks for

---

## Usage: Stage 4

### Setup
```bash
pip install torch "transformers==4.53.2" IndicTransToolkit gTTS pygame
```

**Important:** `transformers` must be exactly `4.53.2` — newer versions break `IndicTransToolkit`'s imports. Install in this order to avoid other packages silently upgrading it.

**Hugging Face authentication required** for first run (model download):
```bash
pip install huggingface_hub
huggingface-cli login
```
Generate a free "Read" token at https://huggingface.co/settings/tokens if you don't have one.

### Run
```bash
python translate.py
```

First run downloads the IndicTrans2 model (~800MB) — cached locally afterward. Requires internet access every run for gTTS speech synthesis (no offline fallback).

### Controls
- `r` — start/stop recording one sign
- `1` / `2` / `3` — switch target language (Hindi / Marathi / Tamil)
- `q` — quit

Same framing tips from Stage 3 apply here.

---

## Project Architecture

```
main.py
├── Shortcuts (imports)
│   └── MediaPipe task classes + running modes
├── extract_landmarks(pose, face, left_hand, right_hand)
│   └── Flattens detected landmarks → 1D array (1692 values)
├── split_hands(hand_result)
│   └── Sorts multiple detected hands into left/right by handedness label
└── main()
    ├── Initialize three landmarkers (Hand, Pose, Face)
    ├── Open webcam (CAP_DSHOW backend for Windows stability)
    ├── Frame loop:
    │   ├── Convert BGR → RGB for MediaPipe
    │   ├── Run all three detectors on frame
    │   ├── Draw landmarks on BGR frame
    │   ├── Optionally record flattened landmarks
    │   └── Handle keypresses (r=record, q=quit)
    └── Cleanup & save
```

---

## Dataset: INCLUDE-50 (AI4Bharat)

The [INCLUDE dataset](https://huggingface.co/datasets/ai4bharat/INCLUDE) from AI4Bharat is a curated collection of ISL sign videos.

**INCLUDE-50 Subset Used:**
- **943 total videos** across 50 ISL word classes
- **Training:** 675 videos (67 per class on average, range 5–233)
- **Validation:** 77 videos (1.5 per class)
- **Test:** 191 videos (3.8 per class)
- **15 categories:** Adjectives, Animals, Clothes, Colours, Days_and_Time, Electronics, Greetings, Home, Jobs, Means_of_Transportation, People, Places, Pronouns, Seasons, Society
- **Filmed in:** Chennai, Tamil Nadu
- **Signers:** Multiple (signer-independent test set means unseen signers in test split)
- **Frame range per clip:** 45–200 frames (~1.5–6.5 seconds at 30 FPS)

**Class Imbalance:** Adjectives dominates with 233 training clips (25% of dataset), while Seasons, Society, and Jobs have only 28–29 clips each. This imbalance is reflected in per-class accuracy variation (100% for well-sampled classes like "16. train ticket", 0% for under-sampled classes like "19. House").

**Regional Note:** ISL varies across India. This dataset focuses on Chennai/Tamil Nadu signing conventions. Future work should include signers from Delhi, Mumbai, Bangalore for geographic coverage.

---

## Known Limitations & Future Work

### Stage 1–4 Constraints (Current)
- **Signer-dependent memorization** — Model achieves 55% val accuracy but only 36.6% test accuracy on unseen signers, indicating it partly learned individual signer characteristics rather than pure sign patterns. This is a known challenge in sign language recognition and requires larger, more diverse training data.
- **Class imbalance** — Adjectives has 233 training clips while Society/Jobs/Electronics have ~29 each. Rare classes have near-zero recall. Balanced sampling or class weighting could help.
- **Limited training data** — 675 examples across 50 classes (13 per class) is below the typically recommended ~100+ per class for deep learning. Data augmentation helped (44% → 55% val) but is no substitute for more real data.
- **Single face/body detection** — script assumes one signer in frame
- **No continuous recognition** — Stage 3/4 requires manual record/stop per sign rather than always-on detection. A continuous sliding-window approach was attempted first but abandoned: the model was trained on isolated, complete sign clips, so classifying arbitrary continuous-video fragments (mid-sign, transitions, stillness) produced near-random predictions. Record-and-classify matches training conditions far more closely.
- **Camera framing sensitivity** — live recognition accuracy is sensitive to how much of the signer's arm span is visible in frame; poor framing causes MediaPipe to extrapolate pose landmarks beyond realistic ranges, degrading prediction quality independent of the model itself.
- **Live accuracy ≈ test accuracy, not higher** — a live demo signer (never seen in training) should expect roughly the same ~1-in-3 success rate as the reported signer-independent test accuracy, even with correct framing and clean recordings.
- **Translation quality depends on recognition accuracy** — Stage 4 translates whatever Stage 3 recognizes; an incorrect sign classification produces a confidently wrong translation. Errors compound rather than self-correct.
- **gTTS requires internet** — no offline speech synthesis fallback currently implemented.
- **Single-word translation only** — IndicTrans2 is used per-recognized-word, not for multi-sign sentence sequences with grammar/context.

### Stage 5 Constraints (Upcoming)
- **No frontend yet** — currently a local Python script with OpenCV window, not a deployable web/mobile interface.
- **No packaged demo** — needs a recorded walkthrough video showing both successes and honest failures for portfolio use.

### Regional Variations & Future Improvements
- Current model trained on Chennai/Tamil Nadu ISL. Signing conventions vary significantly across India (Delhi, Mumbai, Bangalore each have distinct regional styles).
- **Next steps:** Collect signers from multiple regions, retrain with region-balanced data, evaluate geographic generalization.
- **Community involvement:** Partner with Deaf communities in different regions for authentic data collection and validation.

---

## Performance Notes

### Stage 1 — Landmark Extraction
- **FPS:** ~25–30 FPS on CPU (modern Intel i5/i7), all three landmarkers running in parallel
- **Memory footprint:** ~200–300MB (three MediaPipe models loaded)
- **GPU acceleration:** MediaPipe's GPU delegates have documented issues on Windows; CPU performance is sufficient for real-time use

### Stage 2 — Bi-LSTM Training & Inference
- **Training time:** ~4–8 minutes per run on T4 GPU (Colab); ~30–45 minutes on CPU
- **Single inference:** ~2–5ms per 100-frame sequence on GPU, ~15–30ms on CPU
- **Model size:** 1.8M parameters, ~7.2 MB checkpoint file
- **Memory during inference:** ~50–100MB (model weights + batch)

**Hardware Tested:**
- **Stage 1:** Windows 11 with Python 3.12 on Intel i5/i7 CPU, RTX 4050 Laptop GPU
- **Stage 2:** Google Colab T4 GPU, local Windows CPU
- Should work on any machine with Python 3.9+ and a webcam (Stage 1), PyTorch (Stage 2)

### Accuracy Breakdown
- **Test Accuracy (signer-independent):** 36.6% across 50 classes
- **Test Accuracy (by class range):** 0%–100%, median ~77% (classes with >15 training examples)
- **Per-class metrics available:** See confusion_matrix.png and classification_report from Cell 8

---

## File Structure

```
project/
├── Stage 1 — Landmark Extraction
│   ├── main.py    (webcam → landmarks)
│   ├── hand_landmarker.task              (MediaPipe model)
│   ├── pose_landmarker_lite.task         (MediaPipe model)
│   ├── face_landmarker.task              (MediaPipe model)
│   └── landmarks.npy                     (saved sequences from webcam)
│
├── Stage 2 — Training
│   ├── metadata.py                       (verify dataset coverage)
│   ├── download_data.py                  (download INCLUDE-50 from Zenodo)
│   ├── extract_keypoints.py              (process videos → .npy files)
│   ├── ISL.ipynb                         (train model in Colab)
│   ├── videos/                           (943 INCLUDE-50 video clips)
│   ├── extracted_landmarks/              (943 .npy files with landmarks)
│   ├── landmarks_index.csv               (video → label mapping)
│   ├── label_map.csv                     (50 word classes → indices)
│   ├── isl_bilstm_checkpoint.pt          (trained model weights)
│   ├── training_curves_final.png         (loss/accuracy plots)
│   └── confusion_matrix_final.png        (50×50 test set predictions)
│
├── Stage 3 — Real-Time Inference
│   └── inference.py     (webcam → record → classify)
│
├── Stage 4 — Translation & Speech
│   └── translate.py         (record → classify → translate → speak)
│
├── README.md                         (this file)
└── [Stage 5 files coming soon]
```

---

## Troubleshooting

### "No file with name: hand_landmarker.task"
Ensure all three `.task` files are in the same folder as the script. Download links above.

### Camera won't open (stuck on startup)
- Check that no other app (Zoom, Teams, browser) is using the camera
- Try closing and reopening the script
- On Windows: Settings → Privacy & Security → Camera → ensure desktop apps are allowed

### Landmarks not showing / static video
- Face must be visible in frame for all three detectors to work
- Move closer to camera if too far away
- Ensure good lighting

### High memory usage
All three models load on startup (~500MB combined). This is normal and expected.

---

## Contributing & Citation

This project is inspired by:
- Google's MediaPipe Tasks Vision for real-time landmark detection
- AI4Bharat's INCLUDE dataset for Indian Sign Language
- IndicTrans2 for low-resource language translation

If you use this code or dataset, please cite:
```bibtex
@dataset{include_isl,
  title={INCLUDE: Indian Sign Language Dataset},
  author={AI4Bharat},
  url={https://huggingface.co/datasets/ai4bharat/INCLUDE}
}
```

---

## License

MIT License — see LICENSE file for details.
