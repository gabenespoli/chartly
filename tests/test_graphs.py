import warnings

import pandas as pd
import polars as pl
import pytest

from chartly import graphs

warnings.filterwarnings("ignore")


@pytest.fixture
def df_pandas():
    return pd.DataFrame({"cat": ["c", "a", "b"], "val": [1, 2, 3]})


@pytest.fixture
def df_polars():
    return pl.DataFrame({"cat": ["c", "a", "b"], "val": [1, 2, 3]})


def test_category_orders_alphabetical_pandas(df_pandas):
    fig = graphs.graph(df_pandas, x="cat", y="val", graph_type="bar")
    assert list(fig.layout.xaxis.categoryarray) == ["a", "b", "c"]


def test_category_orders_alphabetical_polars(df_polars):
    fig = graphs.graph(df_polars, x="cat", y="val", graph_type="bar")
    assert list(fig.layout.xaxis.categoryarray) == ["a", "b", "c"]


def test_unsortable_column_skips_ordering(capsys):
    df = pd.DataFrame({"cat": ["a", 1, "b"], "val": [1, 2, 3]})
    graphs.graph(df, x="cat", y="val")
    assert "unsortable" in capsys.readouterr().out.lower()
