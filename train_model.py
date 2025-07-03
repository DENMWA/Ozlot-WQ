
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_ml_model(historical_csv, model_output_path="ozlot_ml_model.pkl"):
    """
    Train a Random Forest ML model based on historical draws.

    Args:
        historical_csv (str): Path to historical draws CSV.
        model_output_path (str): Path to save trained ML model.
    """
    df = pd.read_csv(historical_csv)

    if df.shape[0] < 50:
        print("⚠️ Not enough historical draws for reliable training.")
        return

    # Example feature engineering: gaps and sum
    df["Gaps"] = df.apply(lambda row: np.diff(sorted(row.values)), axis=1)
    df["Sum"] = df.sum(axis=1)
    df["Entropy"] = df.apply(lambda row: np.std(row.values), axis=1)

    # Convert list of gaps to multiple features
    gap_features = pd.DataFrame(df["Gaps"].tolist(), columns=[f"Gap{i+1}" for i in range(df["Gaps"].iloc[0].shape[0])])

    # Combine all features
    features = pd.concat([gap_features, df[["Sum", "Entropy"]]], axis=1)
    
    # Dummy target for training (simulate division or win status)
    # You can replace this with real division labels if available.
    df["Target"] = (df["Sum"] % 5 == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(features, df["Target"], test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, model_output_path)
    print(f"Trained model saved to {model_output_path}")

if __name__ == "__main__":
    # Example run
    train_ml_model(historical_csv="historical_draws.csv")
