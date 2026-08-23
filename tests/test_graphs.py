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


def make_shap_frame(n_features: int) -> pd.DataFrame:
    data = {"E[f(x)]": [1.0], "f(x)": [2.0]}
    for i in range(n_features):
        data[f"f{i}"] = [float(i + 1)]
    return pd.DataFrame(data, index=["row1"])


def test_waterfall_counts_other_features_correctly():
    fig = graphs.waterfall(make_shap_frame(6), n_top_features=2)
    labels = [str(x) for x in fig.data[0].y]
    assert "Sum of 4 other features" in labels


def test_waterfall_all_features_requested_adds_no_other_bucket():
    fig = graphs.waterfall(make_shap_frame(4), n_top_features=9)
    labels = [str(x) for x in fig.data[0].y]
    assert len(labels) == 4
    assert not any("other" in label for label in labels)


def make_grouped_stacked_df() -> pd.DataFrame:
    combos = [
        (m, r, s) for m in ["jan", "feb"] for r in ["east", "west"] for s in ["a", "b"]
    ]
    return pd.DataFrame(
        [
            {"month": m, "region": r, "segment": s, "val": (i * 3) % 10}
            for i, (m, r, s) in enumerate(combos)
        ]
    )


def test_grouped_stacked_bar_characterization():
    fig = graphs.graph(
        make_grouped_stacked_df(),
        x="month",
        y="val",
        color="segment",
        bar_group="region",
        barmode="stack",
        graph_type="bar",
    )
    assert str(fig.layout.barmode) == "stack"
    assert list(fig.layout.xaxis.ticktext) == ["feb", "jan"]
    traces = [
        (t.name, t.showlegend, list(t.x), list(t.y), t.marker.color) for t in fig.data
    ]
    assert traces == [
        ("a", True, [-0.2, 0.8], [2, 0], "#636EFA"),
        ("a", False, [0.2, 1.2], [8, 6], "#636EFA"),
        ("b", True, [-0.2, 0.8], [5, 3], "#EF553B"),
        ("b", False, [0.2, 1.2], [1, 9], "#EF553B"),
    ]


def test_add_category_orders_does_not_mutate_kwargs(df_pandas):
    kwargs = {"x": "cat"}
    out = graphs._add_category_orders(df_pandas, ["x"], kwargs)
    assert "category_orders" not in kwargs
    assert out["category_orders"]["cat"] == ["a", "b", "c"]


def test_add_category_orders_preserves_user_order(df_pandas):
    kwargs = {"x": "cat", "category_orders": {"cat": ["c", "a", "b"]}}
    out = graphs._add_category_orders(df_pandas, ["x"], kwargs)
    assert out["category_orders"]["cat"] == ["c", "a", "b"]


def test_sankey_builds_links_and_drops_zero_flows():
    df = pd.DataFrame({"stage1": ["a", "a", "b"], "stage2": ["x", "y", "y"]})
    fig = graphs.sankey(df, node1="stage1", node2="stage2")
    link = fig.data[0].link
    labels = [label.split(" (")[0] for label in fig.data[0].node.label]
    triples = {
        (labels[s], labels[t], v)
        for s, t, v in zip(link.source, link.target, link.value)
    }
    # b -> x has zero rows and must be dropped
    assert triples == {
        ("Total", "a", 2),
        ("Total", "b", 1),
        ("a", "x", 1),
        ("a", "y", 1),
        ("b", "y", 1),
    }
