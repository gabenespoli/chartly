import warnings
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

import chartly.charts as charts
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


def test_compact_renders_header_controls_in_sidebar(monkeypatch):
    df = pl.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    sidebar = MagicMock()
    header = MagicMock()
    monkeypatch.setattr(charts.st, "sidebar", sidebar)
    monkeypatch.setattr(Chart, "header", header)

    Chart(id="compact", data=df, y_opts=["amount"], x_opts=["region"], compact=True)

    header.assert_not_called()
    sidebar.subheader.assert_not_called()
    sidebar.popover.assert_not_called()
    sidebar.expander.assert_called_once_with("Chart Options")
    expander = sidebar.expander.return_value
    sidebar.markdown.assert_called_once()
    assert "padding-top: 0" in sidebar.markdown.call_args.args[0]
    assert sidebar.selectbox.call_count == 0
    assert expander.selectbox.call_count == 10
    expander.empty.assert_called_once()
    expander.empty.return_value.selectbox.assert_called_once()


def test_compact_show_figure_hides_modebar_and_top_margin(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    df = pl.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    sidebar = MagicMock()
    plotly_chart = MagicMock()
    monkeypatch.setattr(charts.st, "sidebar", sidebar)
    monkeypatch.setattr(charts.st, "plotly_chart", plotly_chart)

    chart = Chart(
        id="compact-fig",
        data=df,
        y_opts=["amount"],
        x_opts=["region"],
        compact=True,
    )
    chart.fig = go.Figure()
    chart.show_figure()

    assert plotly_chart.call_args.kwargs["config"] == {
        "displayModeBar": False,
        "scrollZoom": False,
    }
    assert chart.fig.layout.margin.t == 0
    assert chart.fig.layout.margin.b == 140
    assert chart.fig.layout.dragmode is False
    assert chart.fig.layout.xaxis.fixedrange is True
    assert chart.fig.layout.yaxis.fixedrange is True
    assert chart.fig.layout.legend.orientation == "h"
    assert chart.fig.layout.legend.xref == "container"
    assert chart.fig.layout.legend.yref == "container"
    assert chart.fig.layout.legend.yanchor == "bottom"
    assert chart.fig.layout.legend.y == 0.01
    assert chart.fig.layout.legend.xanchor == "center"
    assert chart.fig.layout.legend.x == 0.5


def test_non_compact_show_figure_keeps_default_plotly_config(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    df = pl.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    cols = [MagicMock() for _ in range(5)]
    plotly_chart = MagicMock()
    monkeypatch.setattr(Chart, "header", MagicMock(return_value=cols))
    monkeypatch.setattr(charts.st, "plotly_chart", plotly_chart)

    chart = Chart(id="normal-fig", data=df, y_opts=["amount"], x_opts=["region"])
    chart.fig = go.Figure()
    chart.show_figure()

    assert "config" not in plotly_chart.call_args.kwargs
    assert chart.fig.layout.legend.orientation is None


def test_non_compact_renders_header_controls_in_main_area(monkeypatch):
    df = pl.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    cols = [MagicMock() for _ in range(5)]
    header = MagicMock(return_value=cols)
    sidebar = MagicMock()
    monkeypatch.setattr(charts.st, "sidebar", sidebar)
    monkeypatch.setattr(Chart, "header", header)

    Chart(id="normal", data=df, y_opts=["amount"], x_opts=["region"])

    header.assert_called_once()
    assert sidebar.selectbox.call_count == 0
    sidebar.expander.assert_not_called()
    cols[1].selectbox.assert_called_once()
    cols[2].selectbox.assert_called_once()
    cols[3].selectbox.assert_called_once()
    cols[4].popover.assert_called_once_with("Options")


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


def test_chart_converts_pandas_input():
    df = pd.DataFrame({"region": ["a", "b"], "amount": [1, 2]})
    chart = Chart(id="pdx", data=df, y_opts=["amount"], x_opts=["region"])
    assert isinstance(chart.data, pl.DataFrame)
    chart.update_figure()
    assert chart.fig is not None


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


def test_group_by_date_pandas_input_monthly(date_df_pandas):
    """pandas input is converted at the boundary; months with no rows are
    filled in with zero rows by default."""
    out = Chart.group_by_date(date_df_pandas, "Monthly", date_col="Datetime")
    assert isinstance(out, pl.DataFrame)
    assert list(out["DateGrouping"]) == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert list(out["Amount"]) == [2, 0, 3, 5]


def test_group_by_date_skip_empty_periods(date_df_pandas):
    out = Chart.group_by_date(
        date_df_pandas, "Monthly", date_col="Datetime", show_empty_periods=False
    )
    assert list(out["DateGrouping"]) == ["2024-01", "2024-03", "2024-04"]


def test_group_by_date_fills_quarterly_gap():
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-05", "2024-07-15"]),
            "Amount": [1, 7],
        }
    )
    out = Chart.group_by_date(df, "Quarterly", date_col="Datetime")
    assert list(out["DateGrouping"]) == ["2024-Q1", "2024-Q2", "2024-Q3"]
    assert list(out["Amount"]) == [1, 0, 7]


def test_group_by_date_fills_daily_gap():
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-05", "2024-01-07"]),
            "Amount": [1, 3],
        }
    )
    out = Chart.group_by_date(df, "Daily", date_col="Datetime")
    assert list(out["DateGrouping"]) == ["2024-01-05", "2024-01-06", "2024-01-07"]
    assert list(out["Amount"]) == [1, 0, 3]


def test_group_by_date_fills_weekly_gap():
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-05", "2024-01-19"]),
            "Amount": [1, 3],
        }
    )
    out = Chart.group_by_date(df, "Weekly", date_col="Datetime")
    assert list(out["DateGrouping"]) == ["2024-W01", "2024-W02", "2024-W03"]
    assert list(out["Amount"]) == [1, 0, 3]


def test_group_by_date_fills_biweekly_gap():
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-05", "2024-02-02"]),
            "Amount": [1, 3],
        }
    )
    out = Chart.group_by_date(df, "Bi-Weekly", date_col="Datetime")
    assert list(out["DateGrouping"]) == ["2023-W51", "2024-W01", "2024-W03"]
    assert list(out["Amount"]) == [1, 0, 3]


def test_group_by_date_fills_yearly_gap():
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-01-05", "2026-01-05"]),
            "Amount": [1, 3],
        }
    )
    out = Chart.group_by_date(df, "Yearly", date_col="Datetime")
    assert list(out["DateGrouping"]) == ["2024", "2025", "2026"]
    assert list(out["Amount"]) == [1, 0, 3]


def test_group_by_date_fills_observed_group_combinations(date_df_pandas):
    df = date_df_pandas.copy()
    df["cat"] = ["a", "a", "b", "b"]
    out = Chart.group_by_date(df, "Monthly", date_col="Datetime", grp_col="cat")
    pairs = sorted(zip(out["DateGrouping"], out["cat"], out["Amount"]))
    assert pairs == [
        ("2024-01", "a", 2),
        ("2024-01", "b", 0),
        ("2024-02", "a", 0),
        ("2024-02", "b", 0),
        ("2024-03", "a", 0),
        ("2024-03", "b", 3),
        ("2024-04", "a", 0),
        ("2024-04", "b", 5),
    ]
    assert out["Amount"].sum() == 10


def test_group_by_date_preserves_amount_dtype(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Monthly", date_col="Datetime")
    assert out.schema["Amount"] == pl.Int64
    float_df = pl.DataFrame(
        {
            "Datetime": [date(2024, 1, 5), date(2024, 3, 5)],
            "Amount": [1.5, 2.5],
        }
    )
    out = Chart.group_by_date(float_df, "Monthly")
    assert out.schema["Amount"] == pl.Float64
    assert list(out["Amount"]) == [1.5, 0.0, 2.5]


def test_group_by_date_no_grouping_returns_input_unchanged(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, None)
    assert isinstance(out, pl.DataFrame)
    assert out.shape[0] == 4
    assert "DateGrouping" not in out.columns


def test_group_by_date_empty_frame():
    df = pl.DataFrame(
        {
            "Datetime": pl.Series([], dtype=pl.Date),
            "Amount": pl.Series([], dtype=pl.Float64),
        }
    )
    out = Chart.group_by_date(df, "Monthly")
    assert out.shape[0] == 0


def make_monthly_chart(graph_type="bar", grouped_stacked=False, show_empty=True):
    df = pl.DataFrame(
        {
            "Datetime": [
                date(2024, 1, 5),
                date(2024, 1, 6),
                date(2024, 3, 5),
                date(2024, 3, 6),
            ],
            "Amount": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "grp": ["x", "y", "x", "y"],
        }
    )
    chart = Chart(
        id=f"gap-{graph_type}-{grouped_stacked}-{show_empty}",
        data=df,
        y_opts=["Amount"],
        x_opts=["Datetime"],
        color_opts=["cat", "grp"],
        date_col="Datetime",
        show_empty_periods=show_empty,
    )
    chart.date_grouping = "Monthly"
    chart.min_date_grouping = "2024-01"
    chart.max_date_grouping = "2024-03"
    chart.color = "cat"
    chart.graph_type = graph_type
    if grouped_stacked:
        chart.barmode = "grouped+stacked"
        chart.bar_group = "grp"
    return chart


def test_update_figure_grouped_stacked_shows_empty_period():
    chart = make_monthly_chart(grouped_stacked=True)
    chart.update_figure()
    assert list(chart.fig.layout.xaxis.ticktext) == ["2024-01", "2024-02", "2024-03"]
    for trace in chart.fig.data:
        assert list(trace.y)[1] == 0.0


def test_update_figure_grouped_stacked_hides_empty_periods_when_disabled():
    chart = make_monthly_chart(grouped_stacked=True, show_empty=False)
    chart.update_figure()
    assert list(chart.fig.layout.xaxis.ticktext) == ["2024-01", "2024-03"]


def test_update_figure_bar_shows_empty_period():
    chart = make_monthly_chart()
    chart.update_figure()
    assert list(chart.fig.layout.xaxis.categoryarray) == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]


def test_update_figure_line_shows_empty_period():
    chart = make_monthly_chart(graph_type="line")
    chart.update_figure()
    assert list(chart.fig.layout.xaxis.categoryarray) == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]


def test_group_by_date_pandas_quarterly(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Quarterly", date_col="Datetime")
    assert sorted(out["DateGrouping"].unique()) == ["2024-Q1", "2024-Q2"]


def test_group_by_date_pandas_yearly(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Yearly", date_col="Datetime")
    assert list(out["DateGrouping"].unique()) == ["2024"]


def test_group_by_date_pandas_daily(date_df_pandas):
    out = Chart.group_by_date(date_df_pandas, "Daily", date_col="Datetime")
    assert "2024-01-05" in list(out["DateGrouping"])


def test_group_by_date_quarterly_matches_across_input_types(date_df_pandas):
    df_pl = pl.from_pandas(date_df_pandas)
    out_pl = Chart.group_by_date(df_pl, "Quarterly", date_col="Datetime")
    out_pd = Chart.group_by_date(date_df_pandas, "Quarterly", date_col="Datetime")
    assert sorted(out_pl["DateGrouping"].to_list()) == sorted(
        out_pd["DateGrouping"].to_list()
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


def test_biweekly_even_week_joins_previous_bucket():
    df = pl.DataFrame(
        {
            "Datetime": pd.to_datetime(["2026-12-22", "2026-12-30"]),
            "Amount": [1, 2],
        }
    )
    chart = Chart(id="bwx", data=df, date_col="Datetime")
    chart.date_grouping = "Bi-Weekly"
    chart.add_date_grouping_column()
    # Week 52 is even, so it belongs to the bucket starting at odd week 51
    assert list(chart.data["DateGrouping"]) == ["2026-W51", "2026-W53"]


def test_add_date_grouping_column_quarterly():
    df = pl.DataFrame(
        {
            "Datetime": pd.to_datetime(["2024-02-15", "2024-05-20"]),
            "Amount": [1, 2],
        }
    )
    chart = Chart(id="q", data=df, date_col="Datetime")
    chart.date_grouping = "Quarterly"
    chart.add_date_grouping_column()
    assert list(chart.data["DateGrouping"]) == ["2024-Q1", "2024-Q2"]


def test_get_last_complete_period_weekly_format():
    chart = Chart(id="p", data=pl.DataFrame({"a": [1]}))
    chart.date_grouping = "Weekly"
    assert chart.get_last_complete_period(date(2027, 1, 1)) == "2026-W52"


def test_filter_date_range_inclusive():
    df = pl.DataFrame(
        {"DateGrouping": ["2024-01", "2024-02", "2024-03"], "v": [1, 2, 3]}
    )
    out = Chart._filter_date_range(df, "2024-01", "2024-02")
    assert out["DateGrouping"].to_list() == ["2024-01", "2024-02"]


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
