import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", encoding="utf8")

def get_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}
