from datasets import load_dataset
import pandas as pd

def main():
    print("Loading INCLUDE dataset metadata from hugging face...")
    dataset = load_dataset("ai4bharat/INCLUDE")

    all_rows = []

    for split_name in dataset.keys():
        split_df = dataset[split_name].to_pandas()
        split_df['split'] = split_name
        all_rows.append(split_df)
    df = pd.concat(all_rows, ignore_index=True)

    print(f"Total videos in full dataset {len(df)}")

    include50_df = df[df["include_50"] == True]
    print(f"Videos in INCLLUDE-50 subset: {len(include50_df)}")

    needed_categories = sorted(include50_df["parent_label"].unique())
    print(f"\ncategories needed: {len(needed_categories)}")

    for cat in needed_categories:
        count = len(include50_df[include50_df["parent_label"] == cat])
        print(f" - {cat} : {count} videos")

    include50_df.to_csv("include50_metadata.csv", index=False)
    print("\nsaved filtered metadata to include50_metadata.csv")
    print("Next: Step B will download only the zip files for these categories.")

if __name__ == "__main__":
    main()