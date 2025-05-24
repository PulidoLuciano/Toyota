import dagster as dg
from .utils import load_dataset, get_metrics, LinearRegDiagnostic
from dagstermill import define_dagstermill_asset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from .string_utils import *

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
def load_datasets() -> pd.DataFrame:
    toyota_df = load_dataset("https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv")
    return toyota_df
    
@dg.asset(
    description="Map strings to numbers",
    group_name="data_preprocessing",
    deps=[load_datasets],
)
def map_strings(toyota_df: pd.DataFrame) -> pd.DataFrame:
    toyota = toyota_df.copy()
    
    toyota = apply_string_treatments(toyota, ["Model", "Fuel_Type"])
    toyota = infer_new_model_columns(toyota)
    toyota.drop(columns=["Model", "Fuel_Type"], axis=1, inplace=True)
    return toyota

@dg.asset(
    description="Cut instances from lower and upper bounds in feature selected",
    group_name="data_preprocessing",
)
def toyota_cut(map_strings: pd.DataFrame) -> pd.DataFrame:
    toyota = map_strings.copy()
    toyota = toyota[toyota["cc"] != 16000]
    toyota = toyota[toyota["m_mpv_verso"] != 1]
    toyota.loc[(toyota["m_terra"] == 1) & ((toyota["m_comfort"] == 1) | (toyota["m_sedan"] == 1)), "m_terra"] = 0
    toyota.drop(columns=["Id", "Cylinders", "Petrol", "m_life_months", "Mfg_Month", "Radio_cassette", "Age_08_04", "m_matic", "m_dsl", "m_airco", "valve", "m_mpv_verso",
    "m_keuze_occ_uit", "m_g3", "m_b_ed", "m_sw", "m_xl", "m_pk", "m_nav", "m_ll", "m_gl", "m_comm"], inplace=True)
    return toyota

@dg.asset(
    description="Scale the dataset",
    group_name="data_preprocessing",
    ins={
        "toyota_cut": dg.AssetIn(key=dg.AssetKey("toyota_cut")),
    },
)
def toyota_scale(toyota_cut: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    toyota_normalized = toyota_cut.copy()
    columns = toyota_normalized.columns
    toyota_normalized[columns] = scaler.fit_transform(toyota_normalized[columns])
    toyota_normalized["Price"] = toyota_cut["Price"]
    toyota_normalized = pd.DataFrame(toyota_normalized, columns=columns)
    return toyota_normalized


@dg.asset(
    description="Apply ln transformation to the dataset",
    group_name="data_preprocessing",
)
def ln_transform(toyota_scale: pd.DataFrame) -> pd.DataFrame:
    toyota_transformed = toyota_scale.copy()
    toyota_transformed['KM'] = np.log(toyota_transformed['KM']+1)
    toyota_transformed['KM'] = np.sqrt(toyota_transformed['KM'])
    toyota_transformed['Weight'] = np.log(toyota_transformed['Weight']+1)
    toyota_transformed['Weight'] = np.sqrt(toyota_transformed['Weight'])
    return toyota_transformed

@dg.asset(
    description="Delete features from the dataset",
    group_name="data_preprocessing",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    },
)
def toyota_clean(ln_transform: pd.DataFrame) -> pd.DataFrame:
    columns = ["Central_Lock", "Met_Color", "Airbag_2", "ABS", "Backseat_Divider", "Metallic_Rim", "Radio", "Diesel", "Airbag_1", "Sport_Model", "m_16v", "m_vvti", "Automatic",
               "Gears", "m_sedan", "m_bns", "m_wagon", "Power_Steering", "Mistlamps", "Tow_Bar", "Doors", "m_matic4", "m_matic3", "m_g6", "m_gtsi", "m_sport", "Boardcomputer", 
               "m_terra", "m_luna", "m_sol", "m_comfort", "CD_Player", "Powered_Windows", "BOVAG_Guarantee", "Airco", "Mfr_Guarantee", "m_hatch_b", "m_liftb", "m_d4d"]
    toyota = ln_transform.drop(columns, axis=1)
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
    mlflow.log_params({"n_features": len(toyota_clean.columns) - 1, "n_observations": len(toyota_clean)})

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
        diagnosticPlotter = LinearRegDiagnostic(model)
        diagnosticPlotter()
        mlflow.log_artifact("./images/residual_plots.png")
        mlflow.end_run()
        metrics_all.append(metrics)
    metrics_means = {key: np.mean([metrics[key] for metrics in metrics_all]) for key in metrics_all[0]}
    mlflow.log_metrics(metrics_means)
    return metrics_means


toyota_strings_notebook = define_dagstermill_asset(
    name="toyota_strings_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/toyota_strings.ipynb"),
    group_name="notebook",
    description="Strings analysis of the Toyota dataset",
    ins={
        "toyota_df": dg.AssetIn(key=dg.AssetKey("toyota_df")),
    }
)

ridge_selection_notebook = define_dagstermill_asset(
    name="ridge_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/ridge_selection.ipynb"),
    group_name="notebook",
    description="Ridge selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

lasso_selection_notebook = define_dagstermill_asset(
    name="lasso_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/lasso_selection.ipynb"),
    group_name="notebook",
    description="Lasso selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

sequence_selection_notebook = define_dagstermill_asset(
    name="sequence_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/sequence_selection.ipynb"),
    group_name="notebook",
    description="Sequence selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

pca_notebook = define_dagstermill_asset(
    name="pca_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/pca.ipynb"),
    group_name="notebook",
    description="PCA of the dataset",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

first_data_cleaning_notebook = define_dagstermill_asset(
    name="first_data_cleaning_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/first_data_cleaning.ipynb"),
    group_name="notebook",
    description="First data cleaning of the dataset",
    ins={
        "map_strings": dg.AssetIn(key=dg.AssetKey("map_strings")),
    }
)

manual_feature_selection_notebook = define_dagstermill_asset(
    name="manual_feature_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/manual_feature_selection.ipynb"),
    group_name="notebook",
    description="Manual feature selection of the dataset",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)