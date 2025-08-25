# src/data/data_pipeline.py

import pandas as pd
import os

def preprocess():
    # Step 1: Download Iris dataset from a raw CSV URL
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)

    # Step 2: Basic validation
    assert not df.isnull().values.any(), "Dataset contains null values"

    # Step 3: Output path (LOCAL path, not SageMaker system path)
    output_path = "ml_outputs/processing/train"
    os.makedirs(output_path, exist_ok=True)

    # Step 4: Save to local output path
    df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    print(f"✅ Preprocessing complete! Saved to {output_path}/train.csv")

if __name__ == "__main__":
    preprocess()
