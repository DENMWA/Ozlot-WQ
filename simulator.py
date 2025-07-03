
import pandas as pd
import numpy as np

def simulate_draws(predictions_csv, n_simulations=10000, numbers_per_draw=7, pool_size=47):
    """
    Run large-scale simulations to test prediction performance.

    Args:
        predictions_csv (str): Path to predictions CSV with 'Prediction Set'.
        n_simulations (int): Number of simulated draws.
        numbers_per_draw (int): Numbers drawn per game.
        pool_size (int): Total number pool (default 47 for Oz Lotto).

    Returns:
        pd.DataFrame: Simulation summary per prediction set.
    """
    df = pd.read_csv(predictions_csv)

    if "Prediction Set" not in df.columns:
        raise ValueError("Missing 'Prediction Set' column in predictions CSV.")

    # Parse numbers
    df['Numbers'] = df["Prediction Set"].str.split().apply(lambda x: [int(i) for i in x])

    # Simulation
    results = []
    for _ in range(n_simulations):
        simulated_draw = set(np.random.choice(range(1, pool_size + 1), numbers_per_draw, replace=False))
        for idx, row in df.iterrows():
            match_count = len(simulated_draw.intersection(set(row['Numbers'])))
            results.append({"Set": row["Prediction Set"], "Matches": match_count})

    result_df = pd.DataFrame(results)

    # Aggregate statistics
    summary = result_df.groupby("Set")["Matches"].value_counts().unstack(fill_value=0)
    summary.columns = [f"{col} Matches" for col in summary.columns]

    return summary


if __name__ == "__main__":
    # Example run
    summary = simulate_draws(
        predictions_csv="predictions.csv",
        n_simulations=1000
    )
    summary.to_csv("simulation_results.csv")
    print("Simulation complete. Results saved to simulation_results.csv")
