import warnings

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
