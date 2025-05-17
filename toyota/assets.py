import dagster as dg
from .utils import load_dataset, get_metrics
from dagstermill import define_dagstermill_asset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm

@dg.multi_asset(
    description="Load necessary datasets",
    group_name="data_ingestion",
    outs={
        "toyota_df": dg.AssetOut(
            description="Toyota dataset",
            group_name="data_ingestion",
        ),
    },
)
def load_datasets():
    toyota_df = load_dataset("https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv")
    return toyota_df
    
@dg.asset(
    description="Map strings to numbers",
    group_name="data_preprocessing",
    deps=[load_datasets],
)
def map_strings(toyota_df: pd.DataFrame) -> pd.DataFrame:
    toyota = toyota_df.copy()
    toyota["Fuel_Type"] = toyota["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
    return toyota

@dg.asset(
    description="Apply ln transformation to the dataset",
    group_name="data_preprocessing",
)
def ln_transform(map_strings: pd.DataFrame) -> pd.DataFrame:
    toyota = map_strings.copy()
    columns = []
    for col in columns:
        toyota[col] = np.log(toyota[col])
    return toyota

@dg.asset(
    description="Cut instances from lower and upper bounds in feature selected",
    group_name="data_preprocessing",
)
def toyota_cut(ln_transform: pd.DataFrame) -> pd.DataFrame:
    columns = [] # {name: <name>, lower: <lower_bound>, upper: <upper_bound>}
    for col in columns:
        if col["lower"] is not None:
            ln_transform = ln_transform[ln_transform[col["name"]] >= col["lower"]]
        if col["upper"] is not None:
            ln_transform = ln_transform[ln_transform[col["name"]] <= col["upper"]]
    return ln_transform

@dg.asset(
    description="Delete features from the dataset",
    group_name="data_preprocessing",
    ins={
        "toyota_cut": dg.AssetIn(key=dg.AssetKey("toyota_cut")),
    },
)
def toyota_clean(toyota_cut: pd.DataFrame) -> pd.DataFrame:
    columns = ["Model"]
    toyota = toyota_cut.drop(columns, axis=1)
    return toyota

@dg.multi_asset(
    description="Split the dataset using k-fold cross validation",
    group_name="data_preprocessing",
    outs={
        "train_indexes": dg.AssetOut(
            description="Train indexes",
            group_name="data_preprocessing",
        ),
        "test_indexes": dg.AssetOut(
            description="Test indexes",
            group_name="data_preprocessing",
        ),
    },
    required_resource_keys={"mlflow"},
)
def split_folds(context: dg.AssetExecutionContext, toyota_clean: pd.DataFrame):
    split_params = {
        "n_splits": 5,
        "random_state": 42,
        "shuffle": True,
    }
    kf = KFold(**split_params)
    folds = kf.split(toyota_clean)

    train_indexes = []
    test_indexes = []

    for (train_index, test_index) in folds:
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    mlflow = context.resources.mlflow
    mlflow.set_tag("mlflow.runName", "toyota_runs")
    mlflow.log_params(split_params)

    return train_indexes, test_indexes

@dg.asset(
    description = "Train model with k-fold cross validation",
    group_name="model_training",
    ins={
        "toyota_clean": dg.AssetIn(key=dg.AssetKey("toyota_clean")),
        "train_indexes": dg.AssetIn(key=dg.AssetKey("train_indexes")),
    },
    required_resource_keys={"mlflow"},
)
def train_models(context: dg.AssetExecutionContext, toyota_clean, train_indexes):
    mlflow = context.resources.mlflow
    models = []
    for i, train_index in enumerate(train_indexes):
        train_fold = toyota_clean.iloc[train_index]
        mlflow.start_run(run_name=f"Fold {i}", nested=True)
        mlflow.autolog()
        X_train = sm.add_constant(train_fold.drop(columns=["Price"], axis=1))
        y_train = train_fold["Price"]
        model = sm.OLS(y_train, X_train).fit()
        mlflow.statsmodels.log_model(model, f"linear_regression_model_{i}")
        model_data = {
            "model": model,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        mlflow.end_run()
        models.append(model_data)
    return models

@dg.asset(
    description = "Evaluate model with k-fold cross validation",
    group_name="model_evaluation",
    ins={
        "toyota_clean": dg.AssetIn(key=dg.AssetKey("toyota_clean")),
        "test_indexes": dg.AssetIn(key=dg.AssetKey("test_indexes")),
        "train_models": dg.AssetIn(key=dg.AssetKey("train_models")),
    },
    required_resource_keys={"mlflow"},
)
def evaluate_model(context: dg.AssetExecutionContext, toyota_clean, test_indexes, train_models):
    mlflow = context.resources.mlflow
    metrics_all = []
    for i, test_index in enumerate(test_indexes):
        test_fold = toyota_clean.iloc[test_index]
        mlflow.start_run(run_id=train_models[i]["mlflow_run_id"], nested=True)
        model = train_models[i]["model"]
        X_test = sm.add_constant(test_fold.drop(columns=["Price"], axis=1))
        y_test = test_fold["Price"]
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        metrics_all.append(metrics)
    return metrics_all


toyota_strings_notebook = define_dagstermill_asset(
    name="toyota_strings_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/toyota_strings.ipynb"),
    group_name="raw_data_analysis",
    description="Strings analysis of the Toyota dataset",
    ins={
        "toyota_df": dg.AssetIn(key=dg.AssetKey("toyota_df")),
    }
)