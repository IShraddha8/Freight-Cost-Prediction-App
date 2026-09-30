from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "invoice_flag_model.pkl"
_artifact = joblib.load(MODEL_PATH)


def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """input_data keys: invoice_quantity, invoice_dollars, Freight,
    total_item_quantity, total_item_dollars (each a list with one value)"""
    df = pd.DataFrame(input_data)[_artifact["features"]]
    scaled = _artifact["scaler"].transform(df)
    df["Predicted_Flag"] = _artifact["model"].predict(scaled)
    return df
