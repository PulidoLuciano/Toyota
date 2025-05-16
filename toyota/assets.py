import dagster as dg
from .utils import load_dataset
from dagstermill import define_dagstermill_asset
import pandas as pd
import numpy as np

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
    description="Apply ln transformation to the dataset",
    group_name="data_preprocessing",
    deps=[load_datasets],
)
def ln_transform(toyota_df: pd.DataFrame) -> pd.DataFrame:
    toyota = toyota_df.copy()
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
    columns = []
    toyota = toyota_cut.drop(columns)
    return toyota

toyota_strings_notebook = define_dagstermill_asset(
    name="toyota_strings_notebook",
    notebook_path= dg.file_relative_path(__file__, "./notebooks/toyota_strings.ipynb"),
    group_name="raw_data_analysis",
    description="Strings analysis of the Toyota dataset",
    ins={
        "toyota_df": dg.AssetIn(key=dg.AssetKey("toyota_df")),
    }
)