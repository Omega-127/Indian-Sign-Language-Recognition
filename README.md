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

---

## Technical Stack

### Vision & Landmarks
- **MediaPipe 0.10.35** — Hand, Pose, and Face landmark detection
- **OpenCV** — webcam capture and real-time visualization
- **NumPy** — array operations and efficient storage

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

## Project Architecture

```
stage1_landmark_extraction.py
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

## Dataset: INCLUDE-50

**Stage 2** will use the [INCLUDE dataset](https://huggingface.co/datasets/ai4bharat/INCLUDE) from AI4Bharat:
- **50 everyday ISL words** (greetings, colors, family, places, etc.)
- Multiple signers, multiple takes per sign
- Filmed in Chennai, Tamil Nadu (note: ISL varies regionally across India)
- Available on Hugging Face + GitHub

Early collection in Stage 1 is intentional — recording 5–10 extra examples per sign yourself improves robustness and is a strong portfolio signal ("collected my own data").

---

## Known Limitations & Future Work

### Current Constraints
- **Single face/body detection** — script assumes one signer in frame
- **No temporal smoothing yet** — rapid prediction changes between frames (fixed in Stage 3)
- **Webcam-only input** — video file support coming in Stage 2

### Regional Variations
This project focuses on ISL as spoken in Chennai (INCLUDE dataset origin). ISL is not monolithic — regional variations exist. Future work should include signers from other regions (Delhi, Mumbai, Bangalore) to build a more inclusive model.

---

## Performance Notes

### Inference Speed
- **Landmark extraction:** ~25–30 FPS on CPU (modern laptops)
- **Memory footprint:** ~200–300MB (three models loaded)
- **GPU acceleration:** Untested; MediaPipe's GPU delegates have known issues with HolisticLandmarker, which is why this uses three separate landmarkers

### Hardware Tested
- Windows 11, Python 3.12, Intel i5/i7 processors
- Should work on any machine with Python 3.9+ and a webcam

---

## File Structure

```
project/
├── main.py    (current: real-time webcam)
├── hand_landmarker.task              (downloaded model)
├── pose_landmarker_lite.task         (downloaded model)
├── face_landmarker.task              (downloaded model)
├── landmarks.npy                     (auto-generated: saved sequences)
└── README.md                         (this file)
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

