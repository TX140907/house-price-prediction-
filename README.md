# house-price-prediction-
My pet project

## Model Performance
The model is evaluated using 5-Fold Cross Validation.

- Metric: Root Mean Squared Error (RMSE)
- Target transformation: `log1p(SalePrice)`
- Cross-validation: 5-Fold KFold
- Mean CV RMSE: **0.1289**
### Model

The current model uses XGBoost (`XGBRegressor`) with preprocessing handled by a Scikit-learn pipeline.

Main preprocessing steps:

- Median imputation for missing numerical values
- Missing-category imputation for categorical values
- One-Hot Encoding
- Rare categorical values grouped using `min_frequency=5`

Feature engineering:

- `TotalSF` = basement + first floor + second floor area
- `TotalPorchSF` = total porch/deck area

### Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib
```
### Prepare the dataset

Place the dataset files in your data folder:
```text
train.csv
test.csv
```
Update the paths in `train.py` and `predict.py` if necessary.

### Train the model
```bash
python train.py
```
The script will:
- load the training data
- create engineered features
- preprocess numerical and categorical features
- evaluate the model using 5-Fold Cross Validation
- train the final XGBoost model on the full training dataset
- save the trained pipeline as `house_price_model.joblib`

### Make predictions
```bash
python predict.py
```
The script will:

- load `house_price_model.joblib`
- load the test dataset
- apply the same feature engineering used during training
- predict house prices
- convert predictions back from log scale using `expm1`
- save the final predictions to `predictions.csv`
