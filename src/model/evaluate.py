import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def evaluate():
    # Step 1: Load model and label encoder
    model = joblib.load("ml_outputs/model/model.joblib")
    label_encoder = joblib.load("ml_outputs/model/label_encoder.joblib")

    # Step 2: Load evaluation dataset
    df = pd.read_csv("ml_outputs/processing/train/train.csv")
    X = df.drop("species", axis=1)
    y = label_encoder.transform(df["species"])

    # Step 3: Predict and evaluate
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Step 4: Log and save metrics
    print(f"✅ Model Accuracy: {accuracy:.4f}")

    metrics = {
        "accuracy": accuracy,
        "passed_threshold": accuracy >= 0.9  # Change threshold as needed
    }

    os.makedirs("ml_outputs/metrics", exist_ok=True)
    with open("ml_outputs/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Evaluation metrics saved to ml_outputs/metrics/metrics.json")

if __name__ == "__main__":
    evaluate()
