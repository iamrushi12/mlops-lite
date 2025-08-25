import subprocess
import json
import os

def run_step(name, command):
    print(f"\n🔹 Running step: {name}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(f"❌ Step '{name}' failed!")
    print(f"✅ Step '{name}' completed successfully.")

def evaluate_condition():
    with open("ml_outputs/metrics/metrics.json") as f:
        metrics = json.load(f)
    accuracy = metrics["accuracy"]
    passed = metrics["passed_threshold"]
    print(f"\n📊 Model Accuracy: {accuracy:.4f}")
    if not passed:
        raise Exception("❌ Model did not meet accuracy threshold. Pipeline stopped.")
    print("✅ Model passed evaluation threshold. Ready for deployment!")

def main():
    print("\n🚀 Starting MLOps Pipeline Simulation...")

    run_step("Data Preprocessing", "python src/data/data_pipeline.py")
    run_step("Model Training", "python src/model/train.py")
    run_step("Model Evaluation", "python src/model/evaluate.py")
    evaluate_condition()

    print("\n🎉 MLOps Pipeline Complete!")

if __name__ == "__main__":
    main()
