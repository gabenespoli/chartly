import warnings
from datetime import date

import polars as pl

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


def test_sunburst_without_colormaps():
    chart = make_chart("sunburst")
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
