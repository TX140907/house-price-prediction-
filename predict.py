import pandas as pd
import numpy as np
import joblib

from create_features_func import create_features

DATA_DIR="test.csv"
model = joblib.load("baseline_pipeline.joblib")

df = pd.read_csv(DATA_DIR)
ids=df["Id"]
TARGET="SalePrice"
FEATURES=[col for col in df.columns if col not in ["Id",TARGET]]
X=df[FEATURES].copy()

X = create_features(X)

pred_log = model.predict(X)

pred_price = np.expm1(pred_log)

result = pd.DataFrame({
    "Id": ids,
    "SalePrice": pred_price.astype("int")
})

result.to_csv(
    "predictions.csv",
    index=False
)

print(result.head())