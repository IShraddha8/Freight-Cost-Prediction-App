"""
Run this ONCE on your laptop (where data/inventory.db lives):

    python train_and_save_models.py

It trains both models exactly like your notebooks, saves them into /models,
and writes a requirements.txt pinned to YOUR installed library versions
(so the saved models load correctly on Streamlit Cloud).
"""
import os
import sqlite3
from importlib import metadata

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, f1_score, make_scorer,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Finds your database automatically in either folder
DB_PATH = next((p for p in ["data/inventory.db", "DATABASE/inventory.db"] if os.path.exists(p)), None)
if DB_PATH is None:
    raise SystemExit("inventory.db not found. Put it inside the 'data' folder and run again.")
FREIGHT_FEATURES = ["Quantity", "Dollars"]   # use ["Dollars"] to match your notebook exactly
USE_GRID_SEARCH = True                 # False = faster, plain RandomForest

os.makedirs("models", exist_ok=True)
conn = sqlite3.connect(DB_PATH)

# ------------------------------------------------------------------
# 1) Freight cost model (Linear Regression)
# ------------------------------------------------------------------
vendor_df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
X = vendor_df[FREIGHT_FEATURES]
y = vendor_df["Freight"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
freight_model = LinearRegression().fit(X_train, y_train)

pred = freight_model.predict(X_test)
print(f"[Freight] MAE={mean_absolute_error(y_test, pred):.2f}  R2={r2_score(y_test, pred)*100:.2f}%")

joblib.dump({"model": freight_model, "features": FREIGHT_FEATURES}, "models/freight_model.pkl")

# ------------------------------------------------------------------
# 2) Invoice flag model (Random Forest)
# ------------------------------------------------------------------
df = pd.read_sql_query("""
WITH purchase_agg AS (
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand) AS total_brands,
        SUM(p.Quantity) AS total_item_quantity,
        SUM(p.Dollars)  AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
)
SELECT
    vi.PONumber,
    vi.Quantity AS invoice_quantity,
    vi.Dollars  AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
FROM vendor_invoice vi
LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber
""", conn)


def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0


df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)

FLAG_FEATURES = ["invoice_quantity", "invoice_dollars", "Freight",
                 "total_item_quantity", "total_item_dollars"]
X = df[FLAG_FEATURES]
y = df["flag_invoice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

if USE_GRID_SEARCH:
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }
    search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                          param_grid, scoring=make_scorer(f1_score), cv=5, verbose=1, n_jobs=-1)
    search.fit(X_train_s, y_train)
    flag_model = search.best_estimator_
    print("[Flag] best params:", search.best_params_)
else:
    flag_model = RandomForestClassifier(random_state=42).fit(X_train_s, y_train)

print("[Flag] test performance:")
print(classification_report(y_test, flag_model.predict(X_test_s)))

joblib.dump({"model": flag_model, "scaler": scaler, "features": FLAG_FEATURES},
            "models/invoice_flag_model.pkl")

# ------------------------------------------------------------------
# 3) requirements.txt pinned to your installed versions
# ------------------------------------------------------------------
lines = []
for pkg in ["streamlit", "pandas", "numpy", "plotly", "scikit-learn", "joblib"]:
    try:
        lines.append(f"{pkg}=={metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        lines.append(pkg)
with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\nDone! Saved models/freight_model.pkl, models/invoice_flag_model.pkl and requirements.txt")
