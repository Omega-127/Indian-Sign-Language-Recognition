from datasets import load_datasets
import pandas as pd

def main():
    print("Loading INCLUDE dataset metadata from hugging face...")
    dataset = load_datasets("ai4baharat/INCLUDE")

    all_rows = []

    for split_name in dataset.keys():
        