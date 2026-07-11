import torch
import numpy as np
import pandas as pd
from stage3 import BiLSTMclassifier, classify_clip

# Reuse the same model class and classify_clip from your Stage 3 script
# (copy-paste those two, or import if you split into modules)

checkpoint = torch.load("isl_bilstm_checkpoint.pt", map_location="cpu")
idx_to_label = {int(k): v for k, v in checkpoint["idx_to_label"].items()}

model = BiLSTMclassifier(
    input_size=checkpoint.get("input_size", 1692),
    hidden_size=checkpoint.get("hidden_size", 128),
    num_layers=checkpoint.get("num_layers", 1),
    num_classes=checkpoint["num_classes"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Pick a few known "Thank you" clips from your test set
index_df = pd.read_csv("landmarks_index.csv")
thank_you_clips = index_df[index_df["label"] == "55. Thank you"]
print(thank_you_clips[["npy_path", "split"]])

# Test one
sample = thank_you_clips.iloc[0]
frames = np.load(sample["npy_path"])
print(f"Frames: {len(frames)}, split: {sample['split']}")

results = classify_clip(model, frames, torch.device("cpu"), idx_to_label, top_k=3)
print("Predictions on REAL training clip:")
for label, prob in results:
    print(f"  {label:<25} {prob:.1%}")

# Same setup as before, just testing a different, very strong class
train_ticket_clips = index_df[index_df["label"] == "16. train ticket"]
print(train_ticket_clips[["npy_path", "split"]])

# Pick a TEST split one specifically (matches what Cell 8 actually evaluated)
test_clip = train_ticket_clips[train_ticket_clips["split"] == "test"].iloc[0]
frames = np.load(test_clip["npy_path"])
print(f"Frames: {len(frames)}, split: {test_clip['split']}")

results = classify_clip(model, frames, torch.device("cpu"), idx_to_label, top_k=3)
print("Predictions on real TEST clip (16. train ticket):")
for label, prob in results:
    print(f"  {label:<25} {prob:.1%}")