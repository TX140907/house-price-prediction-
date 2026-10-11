import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from create_features_func import create_features

DATA_DIR="F:/data/house_price"
train_df=pd.read_csv(f"{DATA_DIR}/train.csv")

TARGET="SalePrice"
FEATURES=[col for col in train_df.columns if col not in ["Id",TARGET]]

X=train_df[FEATURES].copy()
X = create_features(X)
y=np.log1p(train_df[TARGET].copy())

categorical_features=X.select_dtypes(include=['str','object','category']).columns.tolist()
numerical_features=[col for col in X.columns if col not in categorical_features]

numeric_pipeline= Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="constant",fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore",min_frequency=5))
])
preprocessor=ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
model=XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

baseline_pipeline=Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    baseline_pipeline,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

rmse_scores = -scores

print("RMSE từng fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Std RMSE:", rmse_scores.std())

baseline_pipeline.fit(X,y)
joblib.dump(baseline_pipeline,Path("baseline_pipeline.joblib"))
