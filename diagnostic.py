import numpy as np
import pandas as pd

df = pd.read_csv("landmarks_index.csv")
print("Index rows:", len(df))
print("Columns:", list(df.columns))
print()
print("Split breakdown:")
print(df["split"].value_counts())
print()

# Check a few random .npy files
for i in [0, 100, 500, 942]:
    row = df.iloc[i]
    data = np.load(row["npy_path"])
    print(f"[{i}] {row['label'][:20]:<20} shape: {data.shape}  class: {row['class_index']}")