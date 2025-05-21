import dagster as dg
from .utils import load_dataset, get_metrics, LinearRegDiagnostic
from dagstermill import define_dagstermill_asset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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
    
    toyota = apply_string_treatments(toyota, ["Model", "Fuel_Type"])
    toyota = infer_new_model_columns(toyota)
    toyota.drop(columns=["Model"], axis=1, inplace=True)
    return toyota

@dg.asset(
    description="Cut instances from lower and upper bounds in feature selected",
    group_name="data_preprocessing",
)
def toyota_cut(map_strings: pd.DataFrame) -> pd.DataFrame:
    toyota = map_strings.copy()
    #ln_transform["cc"] = ln_transform["cc"].apply(lambda x: 1600 if x == 16000 else x)
    columns = [
        #{
        #    "name": "Price",
        #    "upper": 30000,
        #    "lower": 0
        #},
        #{
        #    "name": "Weight",
        #    "upper": 1250,
        #    "lower": 0
        #},
        #{
        #    "name": "Guarantee_Period",
        #    "upper": 15,
        #    "lower": 0
        #}
    ] # {name: <name>, lower: <lower_bound>, upper: <upper_bound>}
    for col in columns:
        if col["lower"] is not None:
            toyota = toyota[toyota[col["name"]] >= col["lower"]]
        if col["upper"] is not None:
            toyota = toyota[toyota[col["name"]] <= col["upper"]]
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
    toyota = toyota_cut.drop(columns=["Price"], axis=1)
    columns = toyota.columns
    scaler = MinMaxScaler()
    toyota = scaler.fit_transform(toyota)
    toyota = pd.DataFrame(toyota, columns=columns)
    toyota["Price"] = toyota_cut["Price"]
    return toyota

@dg.asset(
    description="Apply ln transformation to the dataset",
    group_name="data_preprocessing",
)
def ln_transform(toyota_scale: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for col in columns:
        toyota_scale[col] = np.log(toyota_scale[col])
    return toyota_scale

@dg.asset(
    description="Delete features from the dataset",
    group_name="data_preprocessing",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    },
)
def toyota_clean(ln_transform: pd.DataFrame) -> pd.DataFrame:
    columns = ["Id", "Cylinders", "m_matic", "m_matic3", "m_matic4", "Radio_cassette",
                "m_dsl", "m_sport", "m_16v", "Central_Lock", "Met_Color", "Airbag_1", "Airbag_2", 
                "Power_Steering", "Backseat_Divider", "Radio", "Mfg_Month", "m_life_months"
                ,"m_hatch_b", "m_liftb", "Petrol", "Diesel", "m_g6", "m_vvti",
                "m_airco", "m_terra", "m_wagon", "m_luna", "m_sol", "Mistlamps", "valve", "m_sedan", "Sport_Model", "Metallic_Rim",
                "Boardcomputer", "cc", "Airco", "Tow_Bar", "Gears", "ABS", "CD_Player", "Automatic", 
                "Mfr_Guarantee", "Doors", "Age_08_04", "m_d4d", "CNG", "m_comfort", "Quarterly_Tax"]
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
    group_name="exploratory_data_analysis",
    description="Strings analysis of the Toyota dataset",
    ins={
        "toyota_df": dg.AssetIn(key=dg.AssetKey("toyota_df")),
    }
)

toyota_clean_eda_notebook = define_dagstermill_asset(
    name="toyota_clean_eda_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/toyota_clean_eda.ipynb"),
    group_name="exploratory_data_analysis",
    description="Exploratory data analysis of the cleaned Toyota dataset",
    ins={
        "toyota_cut": dg.AssetIn(key=dg.AssetKey("toyota_cut")),
    }
)

ridge_selection_notebook = define_dagstermill_asset(
    name="ridge_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/ridge_selection.ipynb"),
    group_name="model_training",
    description="Ridge selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

lasso_selection_notebook = define_dagstermill_asset(
    name="lasso_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/lasso_selection.ipynb"),
    group_name="model_training",
    description="Lasso selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

sequence_selection_notebook = define_dagstermill_asset(
    name="sequence_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/sequence_selection.ipynb"),
    group_name="model_training",
    description="Sequence selection of the best model",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)

pca_notebook = define_dagstermill_asset(
    name="pca_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/pca.ipynb"),
    group_name="model_training",
    description="PCA of the dataset",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    }
)