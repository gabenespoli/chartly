import warnings
from datetime import date
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from chartly import Chart

warnings.filterwarnings("ignore")


def make_pandas_chart_stub(df: pd.DataFrame) -> SimpleNamespace:
    """Chart construction does not support pandas frames, so exercise the
    pandas branches of instance methods through a stand-in carrying the
    attributes those methods read."""
    return SimpleNamespace(data=df, date_col="Datetime")


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
    with pytest.raises(TypeError):
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


@pytest.fixture
def iso_boundary_df():
    # Both dates are in ISO week 53 of ISO year 2026, though 2027-01-01 has
    # calendar year 2027
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2026-12-30", "2027-01-01"]),
            "Amount": [1, 2],
        }
    )


def test_weekly_labels_use_iso_year_pandas(iso_boundary_df):
    chart = make_pandas_chart_stub(iso_boundary_df)
    chart.date_grouping = "Weekly"
    Chart.add_date_grouping_column(chart)
    assert list(chart.data["DateGrouping"]) == ["2026-W53", "2026-W53"]


def test_weekly_labels_use_iso_year_polars(iso_boundary_df):
    chart = Chart(id="w", data=pl.from_pandas(iso_boundary_df), date_col="Datetime")
    chart.date_grouping = "Weekly"
    chart.add_date_grouping_column()
    assert list(chart.data["DateGrouping"]) == ["2026-W53", "2026-W53"]


def test_biweekly_labels_use_iso_year_polars(iso_boundary_df):
    chart = Chart(id="bw", data=pl.from_pandas(iso_boundary_df), date_col="Datetime")
    chart.date_grouping = "Bi-Weekly"
    chart.add_date_grouping_column()
    # ISO week 53 is odd, so it forms its own bucket labeled with its ISO year
    assert list(chart.data["DateGrouping"]) == ["2026-W53", "2026-W53"]


def test_biweekly_labels_match_between_engines():
    df_pd = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2026-12-22", "2026-12-30"]),
            "Amount": [1, 2],
        }
    )
    stub = make_pandas_chart_stub(df_pd)
    stub.date_grouping = "Bi-Weekly"
    Chart.add_date_grouping_column(stub)
    df_pl = pl.from_pandas(df_pd)
    chart = Chart(id="bwx", data=df_pl, date_col="Datetime")
    chart.date_grouping = "Bi-Weekly"
    chart.add_date_grouping_column()
    # Week 52 is even, so it belongs to the bucket starting at odd week 51
    assert list(stub.data["DateGrouping"]) == ["2026-W51", "2026-W53"]
    assert list(stub.data["DateGrouping"]) == list(chart.data["DateGrouping"])


def test_add_date_grouping_column_quarterly_pandas(iso_boundary_df):
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-02-15", "2024-05-20"]),
            "Amount": [1, 2],
        }
    )
    chart = make_pandas_chart_stub(df)
    chart.date_grouping = "Quarterly"
    Chart.add_date_grouping_column(chart)
    assert list(chart.data["DateGrouping"]) == ["2024-Q1", "2024-Q2"]


def test_get_last_complete_period_weekly_format():
    chart = Chart(id="p", data=pl.DataFrame({"a": [1]}))
    chart.date_grouping = "Weekly"
    assert chart.get_last_complete_period(date(2027, 1, 1)) == "2026-W52"


@pytest.mark.parametrize("frame", ["pandas", "polars"])
def test_filter_date_range_inclusive(frame):
    labels = ["2024-01", "2024-02", "2024-03"]
    if frame == "pandas":
        df = pd.DataFrame({"DateGrouping": labels, "v": [1, 2, 3]})
    else:
        df = pl.DataFrame({"DateGrouping": labels, "v": [1, 2, 3]})
    out = Chart._filter_date_range(df, "2024-01", "2024-02")
    got = (
        out["DateGrouping"].tolist()
        if frame == "pandas"
        else out["DateGrouping"].to_list()
    )
    assert got == ["2024-01", "2024-02"]


def test_highlight_monthly_regions_annotates_values_and_diffs():
    go = pytest.importorskip("plotly.graph_objects")
    months = [date(2023, m, 15) for m in range(7, 13)] + [
        date(2024, m, 15) for m in range(1, 6)
    ]
    df = pl.DataFrame({"Month": months, "amount": [100.0] * len(months)})
    stub = SimpleNamespace(graph_type="bar", data=df, y="amount", fig=go.Figure())
    Chart.highlight_monthly_regions(
        stub,
        grouping="Fiscal Half-Year",
        min_date=date(2022, 6, 1),
        max_date=date(2024, 5, 31),
        min_month_chart=date(2023, 6, 1),
    )
    texts = [a.text for a in stub.fig.layout.annotations]
    # Partial first period Jul-Sep 2023, then Oct 2023 - Mar 2024, then the
    # incomplete final period Apr-May 2024
    assert "300.0" in texts
    assert "600.0" in texts
    assert "200.0" in texts
    assert "+300.0 | +100.0%" in texts
    assert "-400.0 | -66.7%" in texts
