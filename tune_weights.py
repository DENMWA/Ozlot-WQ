
import json
import numpy as np
import pandas as pd

def tune_weights(predictions_csv, weights_json, output_json=None, learning_rate=0.05):
    """
    Auto-tune lambda weights based on prediction results and ML scores.
    
    Args:
        predictions_csv (str): Path to CSV containing predictions with ML Score and Ξ-Score
        weights_json (str): Path to existing weights.json file
        output_json (str): Path to save updated weights (defaults to weights_json)
        learning_rate (float): Adjustment rate for tuning
    """
    # Load predictions and weights
    df = pd.read_csv(predictions_csv)
    with open(weights_json, "r") as f:
        weights = json.load(f)
    
    # Identify lambda columns
    lambda_keys = sorted([key for key in weights.keys() if key.startswith("λ")], key=lambda x: int(x[1:]))
    
    # Aggregate weighted score contributions
    for lam in lambda_keys:
        col_name = lam if lam in df.columns else None
        if col_name:
            # Example heuristic: correlation between lambda feature and Ξ-Score
            correlation = df[[col_name, "Ξ-Score"]].corr().iloc[0, 1]
            if pd.notna(correlation):
                # Adjust weight slightly based on correlation
                adjustment = learning_rate * correlation
                weights[lam] += adjustment
                # Optional: Clamp weights to reasonable bounds
                weights[lam] = max(0.1, min(weights[lam], 5.0))
    
    # Save updated weights
    output_path = output_json if output_json else weights_json
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)
    
    print(f"Weights successfully tuned and saved to {output_path}")

if __name__ == "__main__":
    # Example standalone run (adjust file paths accordingly)
    tune_weights(
        predictions_csv="predictions.csv",
        weights_json="weights.json"
    )
