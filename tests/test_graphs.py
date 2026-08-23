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


def test_unsortable_column_skips_ordering():
    df = pd.DataFrame({"cat": ["a", 1, "b"], "val": [1, 2, 3]})
    with pytest.warns(UserWarning, match="unsortable"):
        graphs.graph(df, x="cat", y="val")


def test_text_auto_enabled_by_default(df_pandas):
    fig = graphs.graph(df_pandas, x="cat", y="val")
    assert fig.data[0].texttemplate is not None


def test_text_auto_false_disables_labels(df_pandas):
    fig = graphs.graph(df_pandas, x="cat", y="val", text_auto=False)
    assert fig.data[0].texttemplate in (None, "")


@pytest.mark.parametrize("graph_type", ["line", "scatter"])
def test_line_and_scatter_work_with_pandas(df_pandas, graph_type):
    fig = graphs.graph(df_pandas, x="cat", y="val", graph_type=graph_type)
    assert fig is not None


def test_get_geo_info_none_defaults_to_world():
    assert graphs.get_geo_info(None)["scope"] == "world"


def test_get_geo_info_unknown_country_raises():
    with pytest.raises(ValueError, match="USA"):
        graphs.get_geo_info("USA")


def test_get_geo_info_known_country():
    assert graphs.get_geo_info("CA")["scope"] == "north america"
