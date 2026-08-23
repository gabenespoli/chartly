import warnings

import streamlit as st

from chartly import Filter

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