# src/model/train.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train():
    # Step 1: Load training data from local preprocessed path
    input_path = "ml_outputs/processing/train/train.csv"
    df = pd.read_csv(input_path)

    # Step 2: Prepare features and target
    X = df.drop("species", axis=1)
    y = df["species"]

    # Step 3: Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y_encoded)

    # Step 5: Save model and encoder locally
    model_dir = "ml_outputs/model"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

    print(f"✅ Model and label encoder saved to {model_dir}/")

if __name__ == "__main__":
    train()
