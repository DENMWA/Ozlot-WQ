
import pandas as pd

def analyze_confidence(predictions_csv, output_csv=None, high_thresh=0.85, low_thresh=0.55):
    """
    Analyze confidence of predictions based on ML Score, Entropy, and Ξ-Score.

    Args:
        predictions_csv (str): Path to predictions CSV with ML Score, Entropy, and Ξ-Score.
        output_csv (str, optional): File to save the results.
        high_thresh (float): High-confidence threshold.
        low_thresh (float): Low-confidence threshold.

    Returns:
        pd.DataFrame: DataFrame with added Confidence Level.
    """
    df = pd.read_csv(predictions_csv)

    if not all(col in df.columns for col in ["ML Score", "Entropy", "Ξ-Score"]):
        raise ValueError("Missing required columns in the predictions CSV.")

    # Normalize features for combined confidence index
    ml_score_norm = (df["ML Score"] - df["ML Score"].min()) / (df["ML Score"].max() - df["ML Score"].min())
    entropy_norm = (df["Entropy"] - df["Entropy"].min()) / (df["Entropy"].max() - df["Entropy"].min())
    xi_score_norm = (df["Ξ-Score"] - df["Ξ-Score"].min()) / (df["Ξ-Score"].max() - df["Ξ-Score"].min())

    # Compute combined confidence index
    df["Confidence Index"] = (ml_score_norm + entropy_norm + xi_score_norm) / 3

    # Assign Confidence Levels
    df["Confidence Level"] = pd.cut(
        df["Confidence Index"],
        bins=[-float("inf"), low_thresh, high_thresh, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Confidence analysis saved to {output_csv}")

    return df

if __name__ == "__main__":
    # Example run
    analyze_confidence(
        predictions_csv="predictions.csv",
        output_csv="confidence_results.csv"
    )
