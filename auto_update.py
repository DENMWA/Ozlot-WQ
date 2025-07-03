
import pandas as pd
import os
import json

def update_draw_record(latest_numbers, historical_csv, output_csv=None):
    """
    Append the latest winning numbers to the historical record.

    Args:
        latest_numbers (list of int): Winning numbers to add.
        historical_csv (str): Path to historical draws CSV.
        output_csv (str, optional): Where to save updated file.
    """
    # Load existing draws
    if os.path.exists(historical_csv):
        history = pd.read_csv(historical_csv)
    else:
        history = pd.DataFrame(columns=["Number 1", "Number 2", "Number 3", "Number 4", "Number 5", "Number 6", "Number 7"])

    # Append new draw
    new_row = pd.Series(latest_numbers, index=history.columns)
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    # Save file
    save_path = output_csv if output_csv else historical_csv
    history.to_csv(save_path, index=False)

    print(f"Draw record updated. Total draws: {len(history)}")

if __name__ == "__main__":
    # Example manual entry (adjust numbers & paths accordingly)
    latest_numbers = [2, 16, 26, 28, 32, 38, 43]  # Enter manually
    update_draw_record(
        latest_numbers=latest_numbers,
        historical_csv="historical_draws.csv"
    )
