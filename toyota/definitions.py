from dagster import Definitions, load_assets_from_modules
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from toyota import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager()
    }
)
