import pandas as pd

df = pd.read_csv("landmarks_index.csv")

# Recover the 4 rows where class index was stored in label column by mistake
repair = {
    1:  "1. loud",
    5:  "19. House",
    41: "83. big large",
    46: "91. Priest"
}

# Fix rows where label is a number (bad rows)
for class_idx, correct_label in repair.items():
    mask = df["label"] == str(class_idx)
    df.loc[mask, "label"] = correct_label
    df.loc[mask, "class_index"] = class_idx

# Verify
print("NaN class_index rows:", df["class_index"].isna().sum())
print("Fixed rows:")
print(df[df["label"].isin(repair.values())][["label", "class_index"]].head(10))

df.to_csv("landmarks_index.csv", index=False)
print("Saved fixed landmarks_index.csv")