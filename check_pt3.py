import os
import pandas as pd

# Define paths to results.csv for each model
models_results = {
    "hammer": r"C:\\Users\\415-28\\Desktop\\dongsup\\runs\\detect\\hammer_training\\results.csv",
    "knife": r"C:\\Users\\415-28\\Desktop\\dongsup\\runs\\detect\\knife_training\\results.csv",
    "gun": r"C:\\Users\\415-28\\Desktop\\dongsup\\runs\\detect\\gun_training\\results.csv",
    "bat": r"C:\\Users\\415-28\\Desktop\\dongsup\\runs\\detect\\bat_training\\results.csv",
    "axe": r"C:\\Users\\415-28\\Desktop\\dongsup\\runs\\detect\\axe_training\\results.csv",
}

def read_results(file_path, model_name):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        print(f"\n=== Metrics for {model_name} (Last Epoch) ===")
        print(f"Epoch: {data['epoch'].iloc[-1]}")
        print(f"Precision: {data['metrics/precision(B)'].iloc[-1]:.4f}")
        print(f"Recall: {data['metrics/recall(B)'].iloc[-1]:.4f}")
        print(f"mAP50: {data['metrics/mAP50(B)'].iloc[-1]:.4f}")
        print(f"mAP50-95: {data['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
        print(f"Validation Box Loss: {data['val/box_loss'].iloc[-1]:.4f}")
        print(f"Validation Class Loss: {data['val/cls_loss'].iloc[-1]:.4f}")
        print(f"Validation DFL Loss: {data['val/dfl_loss'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"Error reading results for {model_name}: {e}")

if __name__ == "__main__":
    for model_name, results_path in models_results.items():
        if os.path.exists(results_path):
            read_results(results_path, model_name)
        else:
            print(f"Results file not found for {model_name}: {results_path}")