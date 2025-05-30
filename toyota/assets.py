import dagster as dg
from .utils import load_dataset, get_metrics, LinearRegDiagnostic, coefs_plot, rmse_plot, r2_plot
from dagstermill import define_dagstermill_asset
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from .string_utils import *
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

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
    toyota['Trunk'] = toyota['Doors'].apply(lambda x: 1 if x == 3 or x == 5 else 0)
    toyota['Five_Doors'] = toyota['Doors'].apply(lambda x: 0 if x <= 3 else 1)
    toyota.drop(columns=["Id", "Cylinders", "Petrol", "m_life_months", "Mfg_Month", "Radio_cassette", "Age_08_04", "m_matic", "m_dsl", "m_airco", "valve", "m_mpv_verso",
    "m_keuze_occ_uit", "m_g3", "m_b_ed", "m_sw", "m_xl", "m_pk", "m_nav", "m_ll", "m_gl", "m_comm", 'Doors'], inplace=True)
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
    description="Cut outliers from the dataset",
    group_name="data_preprocessing",
    ins={
        "ln_transform": dg.AssetIn(key=dg.AssetKey("ln_transform")),
    },
)
def cut_outliers(ln_transform: pd.DataFrame) -> pd.DataFrame:
    toyota_transformed = ln_transform.copy()
    outliers_remove_idx = [
        # Run 1
        138,  523, 1058,  601,  141,  171,  147,  221,  192,  393,  166,  191,
        # Run 2
        588, 351, 189, 53, 184, 186, 696, 960, 11, 223, 402, 12, 1435,
        # Run 3
        7, 8, 161, 10, 179, 379, 618, 378, 125, 913,
        # Run 4
        77, 120, 146, 154, 1109, 1047, 119, 13, 185, 796, 380, 115, 237, 187, 68,
    ]
    toyota_transformed = toyota_transformed.drop(outliers_remove_idx)
    return toyota_transformed

@dg.asset(
    description="Delete features from the dataset",
    group_name="manual_feature_selection",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    },
)
def toyota_clean(cut_outliers: pd.DataFrame) -> pd.DataFrame:
    columns = ["Central_Lock", "Met_Color", "Airbag_2", "ABS", "Backseat_Divider", "Metallic_Rim", "Radio", "Diesel", "Airbag_1", "Sport_Model", "m_16v", "m_vvti", "Automatic",
               "Gears", "m_sedan", "m_bns", "m_wagon", "Power_Steering", "Mistlamps", "Tow_Bar", "m_matic4", "m_matic3", "m_g6", "m_gtsi", "m_sport", "Boardcomputer", 
               "m_terra", "m_luna", "m_sol", "m_comfort", "CD_Player", "Powered_Windows", "BOVAG_Guarantee", "Airco", "Mfr_Guarantee", "m_hatch_b", "m_liftb", "m_d4d", "Five_Doors",
               "Trunk", "m_exec"]
    toyota = cut_outliers.drop(columns, axis=1)
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
def split_folds(context: dg.AssetExecutionContext, cut_outliers: pd.DataFrame):
    split_params = {
        "n_splits": 5,
        "random_state": 42,
        "shuffle": True,
    }
    kf = KFold(**split_params)
    folds = kf.split(cut_outliers)

    train_indexes = []
    test_indexes = []

    for (train_index, test_index) in folds:
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes

@dg.asset(
    description = "Train model with k-fold cross validation",
    group_name="manual_feature_selection",
    ins={
        "toyota_clean": dg.AssetIn(key=dg.AssetKey("toyota_clean")),
        "train_indexes": dg.AssetIn(key=dg.AssetKey("train_indexes")),
    },
    required_resource_keys={"mlflow"},
)
def train_models(context: dg.AssetExecutionContext, toyota_clean, train_indexes):
    mlflow = context.resources.mlflow
    mlflow.set_tag("mlflow.runName", "toyota_runs")
    mlflow.log_params({"n_splits": len(train_indexes)})
    mlflow.log_params({"n_features": len(toyota_clean.columns) - 1})
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
    group_name="manual_feature_selection",
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

@dg.asset(
    description = "Sequence selection of the best model",
    group_name="forward_feature_selection",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    },
    required_resource_keys={"mlflow_sequence"},
)
def sequence_selection(context: dg.AssetExecutionContext, cut_outliers):
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    n_features = 18
    mlflow = context.resources.mlflow_sequence
    mlflow.set_tag("mlflow.runName", f"sequence_{n_features}")

    X = cut_outliers.drop(columns=["Price"], axis=1)
    y = cut_outliers["Price"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features, direction="forward")
    sfs.fit(X_train, y_train)
    selected_features = X_train.columns[sfs.get_support()].tolist()
    mlflow.log_params({"selected_features": selected_features, "n_features": n_features})
    toyota_sequence = cut_outliers[selected_features + ["Price"]]
    return toyota_sequence

@dg.asset(
    description = "Train model with k-fold cross validation",
    group_name="forward_feature_selection",
    ins={
        "sequence_selection": dg.AssetIn(key=dg.AssetKey("sequence_selection")),
        "train_indexes": dg.AssetIn(key=dg.AssetKey("train_indexes")),
    },
    required_resource_keys={"mlflow_sequence"},
)
def train_sequence_model(context: dg.AssetExecutionContext, sequence_selection, train_indexes):
    mlflow = context.resources.mlflow_sequence
    models = []
    for i, train_index in enumerate(train_indexes):
        train_fold = sequence_selection.iloc[train_index]
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
    group_name="forward_feature_selection",
    ins={
        "sequence_selection": dg.AssetIn(key=dg.AssetKey("sequence_selection")),
        "test_indexes": dg.AssetIn(key=dg.AssetKey("test_indexes")),
        "train_sequence_model": dg.AssetIn(key=dg.AssetKey("train_sequence_model")),
    },
    required_resource_keys={"mlflow_sequence"},
)
def evaluate_sequence_model(context: dg.AssetExecutionContext, sequence_selection, test_indexes, train_sequence_model):
    mlflow = context.resources.mlflow_sequence
    metrics_all = []
    for i, test_index in enumerate(test_indexes):
        test_fold = sequence_selection.iloc[test_index]
        mlflow.start_run(run_id=train_sequence_model[i]["mlflow_run_id"], nested=True)
        model = train_sequence_model[i]["model"]
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

@dg.asset(
    description = "Test with ridge and multiples alphas",
    group_name="ridge_selection",
    ins={
        "clean_ridge": dg.AssetIn(key=dg.AssetKey("clean_ridge")),
    },
    required_resource_keys={"mlflow_ridge"},
)
def test_ridge(context: dg.AssetExecutionContext, clean_ridge):
    return test_alphas(context.resources.mlflow_ridge, "testing_ridge", clean_ridge, 0, 4, Ridge)

@dg.asset(
    description = "Test with lasso and multiples alphas",
    group_name="lasso_selection",
    ins={
        "clean_lasso": dg.AssetIn(key=dg.AssetKey("clean_lasso")),
    },
    required_resource_keys={"mlflow_lasso"},
)
def test_lasso(context: dg.AssetExecutionContext, clean_lasso):
    return test_alphas(context.resources.mlflow_lasso, "testing_lasso", clean_lasso, 0, 4, Lasso)

@dg.asset(
    description = "Clean for ridge",
    group_name="ridge_selection",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    },
)
def clean_ridge(cut_outliers):
    return cut_outliers

@dg.asset(
    description = "Clean for lasso",
    group_name="lasso_selection",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    },
)
def clean_lasso(cut_outliers):
    return cut_outliers

@dg.asset(
    description = "Train a single model with lasso",
    group_name="lasso_selection",
    ins={
        "clean_lasso": dg.AssetIn(key=dg.AssetKey("clean_lasso")),
        "train_indexes": dg.AssetIn(key=dg.AssetKey("train_indexes")),
    },
    required_resource_keys={"mlflow_lasso"},
)
def train_lasso(context: dg.AssetExecutionContext, clean_lasso, train_indexes):
    alpha = 0.1
    models = []

    mlflow = context.resources.mlflow_lasso
    mlflow.set_tag("mlflow.runName", "single_lasso")
    mlflow.log_params({"alpha": alpha})
    mlflow.log_params({"n_features": len(clean_lasso.columns) - 1})

    for i, train_index in enumerate(train_indexes):
        train_fold = clean_lasso.iloc[train_index]
        mlflow.start_run(run_name=f"Fold {i}", nested=True)
        mlflow.autolog()
        X_train = train_fold.drop(columns=["Price"], axis=1)
        y_train = train_fold["Price"]
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"lasso_model_{i}")
        model_data = {
            "model": model,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        mlflow.log_text(str(pd.Series(model.coef_, index=X_train.columns)), "features.txt")
        mlflow.end_run()
        models.append(model_data)
    return models

@dg.asset(
    description = "Evaluate model with k-fold cross validation",
    group_name="lasso_selection",
    ins={
        "train_lasso": dg.AssetIn(key=dg.AssetKey("train_lasso")),
        "test_indexes": dg.AssetIn(key=dg.AssetKey("test_indexes")),
        "clean_lasso": dg.AssetIn(key=dg.AssetKey("clean_lasso")),
    },
    required_resource_keys={"mlflow_lasso"},
)
def evaluate_lasso(context: dg.AssetExecutionContext, train_lasso, test_indexes, clean_lasso):
    mlflow = context.resources.mlflow_lasso
    metrics_all = []
    for i, test_index in enumerate(test_indexes):
        test_fold = clean_lasso.iloc[test_index]
        mlflow.start_run(run_id=train_lasso[i]["mlflow_run_id"], nested=True)
        model = train_lasso[i]["model"]
        X_test = test_fold.drop(columns=["Price"], axis=1)
        y_test = test_fold["Price"]
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        metrics_all.append(metrics)
    metrics_means = {key: np.mean([metrics[key] for metrics in metrics_all]) for key in metrics_all[0]}
    mlflow.log_metrics(metrics_means)
    return metrics_means


@dg.asset(
    description = "Train a single model with ridge",
    group_name="ridge_selection",
    ins={
        "clean_ridge": dg.AssetIn(key=dg.AssetKey("clean_ridge")),
        "train_indexes": dg.AssetIn(key=dg.AssetKey("train_indexes")),
    },
    required_resource_keys={"mlflow_ridge"},
)
def train_ridge(context: dg.AssetExecutionContext, clean_ridge, train_indexes):
    alpha = 0.1
    models = []

    mlflow = context.resources.mlflow_ridge
    mlflow.set_tag("mlflow.runName", "single_ridge")
    mlflow.log_params({"alpha": alpha})
    mlflow.log_params({"n_features": len(clean_ridge.columns) - 1})

    for i, train_index in enumerate(train_indexes):
        train_fold = clean_ridge.iloc[train_index]
        mlflow.start_run(run_name=f"Fold {i}", nested=True)
        mlflow.autolog()
        X_train = train_fold.drop(columns=["Price"], axis=1)
        y_train = train_fold["Price"]
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"ridge_model_{i}")
        model_data = {
            "model": model,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        mlflow.log_text(str(pd.Series(model.coef_, index=X_train.columns)), "features.txt")
        mlflow.end_run()
        models.append(model_data)
    return models

@dg.asset(
    description = "Evaluate model with k-fold cross validation",
    group_name="ridge_selection",
    ins={
        "train_ridge": dg.AssetIn(key=dg.AssetKey("train_ridge")),
        "test_indexes": dg.AssetIn(key=dg.AssetKey("test_indexes")),
        "clean_ridge": dg.AssetIn(key=dg.AssetKey("clean_ridge")),
    },
    required_resource_keys={"mlflow_ridge"},
)
def evaluate_ridge(context: dg.AssetExecutionContext, train_ridge, test_indexes, clean_ridge):
    mlflow = context.resources.mlflow_ridge
    metrics_all = []
    for i, test_index in enumerate(test_indexes):
        test_fold = clean_ridge.iloc[test_index]
        mlflow.start_run(run_id=train_ridge[i]["mlflow_run_id"], nested=True)
        model = train_ridge[i]["model"]
        X_test = test_fold.drop(columns=["Price"], axis=1)
        y_test = test_fold["Price"]
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        metrics_all.append(metrics)
    metrics_means = {key: np.mean([metrics[key] for metrics in metrics_all]) for key in metrics_all[0]}
    mlflow.log_metrics(metrics_means)
    return metrics_means

@dg.asset(
    description = "Create principal components",
    group_name="pca",
    ins={
        "toyota_clean": dg.AssetIn(key=dg.AssetKey("toyota_clean")),
    },
    required_resource_keys={"mlflow_pca"},
)
def pcr(context: dg.AssetExecutionContext, toyota_clean):
    n_components = 4
    original_df = toyota_clean.copy()
    df = original_df.drop(columns=["Price"], axis=1)
    mlflow = context.resources.mlflow_pca
    mlflow.set_tag("mlflow.runName", "pcr")
    mlflow.log_params({"n_components": n_components, "n_features": len(df.columns)})
    pca = PCA(n_components=n_components)
    toyota_pca = pca.fit_transform(df)
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(variance_explained))],
        'Variance Explained': variance_explained,
        'Cumulative Variance': cumulative_variance
    })
    mlflow.log_text(variance_df.to_string(), "variance_table.txt")
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    # Scree Plot
    plt.subplot(2, 2, 1)
    plt.bar(range(1, len(variance_explained) + 1), variance_explained)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Scree Plot')
    # Cumulative Variance Plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained')
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "pca_plots.png")
    plt.clf()

    X_train, X_test, y_train, y_test = train_test_split(toyota_pca, original_df["Price"], test_size=0.3, random_state=42, shuffle=True)
    mlflow.autolog()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_test, y_pred)
    mlflow.log_metrics(metrics)
    mlflow.end_run()
    return model

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
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    }
)

lasso_selection_notebook = define_dagstermill_asset(
    name="lasso_selection_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/lasso_selection.ipynb"),
    group_name="notebook",
    description="Lasso selection of the best model",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
    }
)

pca_notebook = define_dagstermill_asset(
    name="pca_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/pca.ipynb"),
    group_name="notebook",
    description="PCA of the dataset",
    ins={
        "cut_outliers": dg.AssetIn(key=dg.AssetKey("cut_outliers")),
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

def test_alphas(mlflow, run_name, df, min_alpha, max_alpha, Method):
    alphas = np.logspace(min_alpha, max_alpha, 20)

    coefs = []
    rmse_list = []
    r2_list = []

    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.log_params({"n_features": len(df.columns) - 1})
    mlflow.log_params({"min_alpha": min_alpha})
    mlflow.log_params({"max_alpha": max_alpha})

    X = df.drop(columns=["Price"], axis=1)
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
    for alpha in alphas:
        mlflow.start_run(run_name=f"alpha_{alpha}", nested=True)
        mlflow.log_params({"alpha": alpha})
        mlflow.autolog()
        model = Method(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2_validated = r2_score(y_test, y_pred)
        coefs.append(model.coef_)
        rmse_list.append(np.sqrt(mse))
        r2_list.append(r2_validated)
        mlflow.log_metrics({"mse": mse, "rmse": np.sqrt(mse), "r2_validation": r2_validated})
        mlflow.log_text(str(pd.Series(model.coef_, index=X_train.columns)), "features.txt")
        mlflow.end_run()
    
    alphas = pd.Index(alphas, name="alpha")
    coefs = pd.DataFrame(coefs, index=alphas, columns=[f"{name}" for _, name in enumerate(X_train.columns)])
    mlflow.log_figure(coefs_plot(coefs), "coefficients.png")
    mlflow.log_figure(rmse_plot(alphas, rmse_list), "rmse.png")
    mlflow.log_figure(r2_plot(alphas, r2_list), "r2.png")
    mlflow.end_run()
    return alphas