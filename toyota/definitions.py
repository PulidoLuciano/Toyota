from dagster import Definitions, load_assets_from_modules
from dagster_mlflow import mlflow_tracking
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from toyota import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota",
        }),
        "mlflow_sequence": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_sequence",
        }),
        "mlflow_ridge": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_ridge",
        }),
        "mlflow_lasso": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_lasso",
        }),
        "mlflow_pca": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_pca",
        }),
    }
)
