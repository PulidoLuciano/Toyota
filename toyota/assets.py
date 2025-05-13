import dagster as dg
from .utils import load_dataset
from dagstermill import define_dagstermill_asset

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

toyota_raw_eda_notebook = define_dagstermill_asset(
    name="toyota_raw_eda_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/toyota_raw_eda.ipynb"),
    group_name="raw_eda",
    description="Exploratory Data Analysis of the Toyota dataset",
    ins={
        "toyota_df": dg.AssetIn(key=dg.AssetKey("toyota_df")),
    }
)