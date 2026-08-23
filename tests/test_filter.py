import warnings

import polars as pl
import streamlit as st

from chartly import Filter, filter_data
from chartly.filter import combine_filters

warnings.filterwarnings("ignore")


def make_filter() -> Filter:
    return Filter(id="test")


def test_selectbox_respects_falsy_default(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    f = make_filter()
    f.selectbox(label="Amount", options=[0, 1, 2], default=0)
    assert f.filters["Amount"] == 0


def test_selectbox_stored_falsy_selection_wins_over_default(monkeypatch):
    key = "test_Amount_eq"
    monkeypatch.setattr(st, "session_state", {key: 0})
    f = make_filter()
    f.selectbox(label="Amount", options=[0, 1, 2], default=2)
    assert f.filters["Amount"] == 0


def test_multiselect_cleared_selection_persists(monkeypatch):
    key = "test_Region"
    monkeypatch.setattr(st, "session_state", {key: []})
    f = make_filter()
    f.multiselect(label="Region", options=["a", "b"], default=["a"])
    assert f.filters["Region"] == []


def test_multiselect_default_applied_when_no_selection(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    f = make_filter()
    f.multiselect(label="Region", options=["a", "b"], default=["a"])
    assert f.filters["Region"] == ["a"]


def test_list_builds_sql_list():
    assert Filter.list(["a", "b", "c"]) == "('a','b','c')"


def test_list_escapes_single_quotes():
    assert Filter.list(["o'brien", "smith"]) == "('o''brien','smith')"


def test_combine_filters_merges_metadata(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    f1 = Filter(id="a")
    f1.selectbox(label="Region", options=["x", "y"], col_name="r")
    f2 = Filter(id="b")
    f2.selectbox(
        label="Amount", options=[0, 1, 2], col_name="amt", filter_type="gte"
    )
    combined = combine_filters(f1, f2)
    assert combined.filters == {**f1.filters, **f2.filters}
    assert combined.col_names == {"Region": "r", "Amount": "amt"}
    assert combined.filter_types["Amount"] == "gte"


def test_combined_filters_apply_in_filter_data(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    f1 = Filter(id="a")
    f1.selectbox(label="Region", options=["east", "west"], col_name="region")
    f2 = Filter(id="b")
    f2.selectbox(
        label="Amount",
        options=[0, 100],
        col_name="amount",
        filter_type="gte",
        bypass_option=0,
    )
    df = pl.DataFrame({"region": ["east", "west", "east"], "amount": [5, 5, 5]})
    combined = combine_filters(f1, f2)
    out = filter_data(df, combined)
    assert out.shape[0] == 2
    assert out["region"].to_list() == ["east", "east"]


def test_filter_sql_handles_scalar_and_list_values():
    f = make_filter()
    f.selectbox(label="Region", options=["east", "west"])
    f.multiselect(label="Product", options=["p1", "p2"], default=["p1"])
    assert (
        f.filter_sql()
        == "WHERE Region in ('east') and Product in ('p1')"
    )


def test_filter_sql_skips_empty_multiselect():
    f = make_filter()
    f.multiselect(label="Product", options=["p1", "p2"])
    assert f.filter_sql() == ""