import warnings
from datetime import date

import pandas as pd
import polars as pl
import pytest

from chartly import Chart

warnings.filterwarnings("ignore")


def make_chart(graph_type: str) -> Chart:
    df = pl.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    return Chart(
        id="test",
        data=df,
        y_opts=["amount"],
        x_opts=["region"],
        default_graph_type=graph_type,
    )


@pytest.fixture
def date_df_pandas():
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-03-15", "2024-04-02"]
            ),
            "Amount": [1, 1, 3, 5],
        }
    )


def test_sunburst_without_colormaps():
    chart = make_chart("sunburst")
    chart.update_figure()
    assert chart.fig is not None


def test_chart_requires_data():
    with pytest.raises(ValueError):
        Chart(id="nodata")


def test_highlight_regions_skips_non_bar_charts():
    chart = make_chart("line")
    chart.update_figure()
    chart.highlight_monthly_regions(
        grouping="Year",
        min_date=date(2024, 1, 1),
        max_date=date(2024, 12, 31),
        min_month_chart=date(2024, 1, 15),
    )
    assert not chart.fig.layout.shapes


def test_group_by_date_pandas_monthly(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Monthly", date_col="Datetime")
    # pd.Grouper emits a bucket per month in range, even when empty
    assert list(out["DateGrouping"]) == ["2024-01", "2024-02", "2024-03", "2024-04"]


def test_group_by_date_pandas_quarterly(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Quarterly", date_col="Datetime")
    assert sorted(out["DateGrouping"].unique()) == ["2024-Q1", "2024-Q2"]


def test_group_by_date_pandas_yearly(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Yearly", date_col="Datetime")
    assert list(out["DateGrouping"].unique()) == ["2024"]


def test_group_by_date_pandas_daily(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Daily", date_col="Datetime")
    assert "2024-01-05" in list(out["DateGrouping"])


def test_group_by_date_polars_quarterly_matches_pandas(date_df_pandas):
    df_pl = pl.from_pandas(date_df_pandas)
    out_pl = Chart.group_by_date(df_pl, "Quarterly", date_col="Datetime")
    out_pd = Chart.group_by_date(date_df_pandas, "Quarterly", date_col="Datetime")
    assert sorted(out_pl["DateGrouping"].to_list()) == sorted(
        out_pd["DateGrouping"].tolist()
    )
