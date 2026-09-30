from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "freight_model.pkl"
_artifact = joblib.load(MODEL_PATH)


def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    """input_data example: {"Quantity": [1200], "Dollars": [18500.0]}"""
    df = pd.DataFrame(input_data)[_artifact["features"]]
    preds = _artifact["model"].predict(df).clip(min=0)  # freight can't be negative
    df["Predicted_Freight"] = preds.round(2)
    return df
