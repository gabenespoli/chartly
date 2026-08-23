# Chartly

Utils for [plotly](https://plotly.com/python/) and
[Streamlit](https://docs.streamlit.io/).

## Install

```bash
pip install chartly
```

Requires Python 3.9+ (see `pyproject.toml`). Core dependencies: pandas, polars,
plotly, streamlit, python-dateutil.

## What's inside

- `chartly.graphs` — thin wrappers around plotly express with sensible defaults:
  `graph` (bar/line/scatter, incl. grouped+stacked bars), `donut`, `sunburst`,
  `sankey`, `map`, and `waterfall` (for SHAP-style values).
- `chartly.charts.Chart` — a Streamlit chart widget: renders graph-type/axis/
  color/facet controls in an options popover, optional date grouping and date
  range filtering, then `update_figure()` / `show_figure()`.
- `chartly.filter.Filter` — collect Streamlit selectbox/multiselect widgets into
  reusable filter objects; apply them to polars DataFrames with
  `filter_data`, or combine them with `combine_filters`.

## Quick example

```python
import polars as pl
import streamlit as st
from chartly import Chart, Filter, filter_data

df = pl.read_parquet("sales.parquet")

region = Filter(id="filters")
region.multiselect("Region", df["Region"].unique().to_list())
df = filter_data(df, region)

chart = Chart(id="sales", data=df, y_opts=["Amount"], x_opts=["Month"])
chart.update_figure()
chart.show_figure()
```

## Development

```bash
uv venv
uv pip install -e ".[dev]"
pytest
pre-commit run --all-files   # black + isort
```
