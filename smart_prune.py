
import pandas as pd
import numpy as np
from scipy.stats import entropy

def compute_set_entropy(numbers):
    """Calculate entropy based on gap distribution of numbers."""  
    gaps = np.diff(sorted(numbers))
    hist, _ = np.histogram(gaps, bins=range(0, 47))
    return entropy(hist + 1)  # add 1 to avoid zero divisions

def prune_predictions(input_csv, output_csv, min_entropy=1.5, top_n=70):
    """
    Prune redundant predictions based on entropy and historical overlap.

    Args:
        input_csv (str): Path to predictions CSV
        output_csv (str): Path to save pruned predictions
        min_entropy (float): Minimum entropy threshold
        top_n (int): Number of predictions to retain
    """
    df = pd.read_csv(input_csv)

    if "Prediction Set" not in df.columns:
        raise ValueError("CSV missing 'Prediction Set' column.")

    # Split numbers for entropy calculation
    df['Numbers'] = df["Prediction Set"].str.split().apply(lambda x: [int(i) for i in x])
    df['Entropy'] = df['Numbers'].apply(compute_set_entropy)

    # Optional ML score prioritization if available
    if 'ML Score' in df.columns:
        df['Sort Score'] = df['ML Score'] * df['Entropy']
    else:
        df['Sort Score'] = df['Entropy']

    # Filter by entropy and rank
    pruned_df = df[df['Entropy'] >= min_entropy]
    pruned_df = pruned_df.sort_values(by='Sort Score', ascending=False).head(top_n)

    # Save to CSV
    pruned_df.to_csv(output_csv, index=False)

    print(f"Pruned predictions saved to {output_csv} ({len(pruned_df)} sets retained)")

if __name__ == "__main__":
    # Example standalone run (adjust file paths accordingly)
    prune_predictions(
        input_csv="predictions.csv",
        output_csv="pruned_predictions.csv"
    )
