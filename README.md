# Chartly

**One DataFrame. Every angle.**

Chartly is a Python library that wraps [Plotly](https://plotly.com/python/) and [Streamlit](https://streamlit.io/) to give you interactive, multi-perspective data visualization with minimal code. Hand it a DataFrame, choose your fields, set your defaults, and explore your data from many angles — all in one chart widget.

---

## Why Chartly?

| Problem | Chartly's Solution |
|---------|-------------------|
| Writing repetitive Plotly boilerplate for each view | One `Chart()` call renders dropdowns for graph type, axes, color, facets, size, and more |
| Switching between bar, line, scatter, donut, sunburst, and map views | Users toggle between all chart types from a single widget — no code changes needed |
| Managing date aggregation (daily → weekly → monthly → quarterly → yearly) | Built-in date grouping selector with automatic aggregation and range filtering |
| Building filter UIs that stay in sync with your data | `Filter` class renders Streamlit widgets and applies them to Polars DataFrames with caching |
| Maintaining consistent color schemes across views | Pass a `colormaps` dict once — colors stay consistent across every chart type and legend |

---

## Quickstart

### Install

```bash
pip install chartly
```

Requires Python 3.7+ (see `pyproject.toml`). Core dependencies: pandas, polars,
plotly, streamlit, python-dateutil.

Internals are polars-only for speed: pass either a polars or a pandas DataFrame
to any public entry point (`Chart`, `graphs.*`, `filter_data`) — pandas input is
converted once at the boundary, and everything downstream runs in polars.

### Minimal Example

```python
import polars as pl
import streamlit as st
from chartly import Chart

df = pl.read_csv("sales.csv")

chart = Chart(
    id="revenue",
    data=df,
    default_x="Region",
    default_y="Revenue",
    default_color="Product",
    default_graph_type="bar",
)
chart.update_figure()
chart.show_figure()
```

That's it. Your users now have interactive dropdowns to:
- Switch between **bar, line, scatter, donut, sunburst**, and **map** views
- Change x/y axes and color encoding on the fly
- Add facet splits (rows and columns)
- Toggle bar modes (grouped, stacked, overlaid, grouped+stacked)
- Adjust chart height and orientation

### With Filters

```python
from chartly import Chart, Filter, filter_data

# Create interactive sidebar filters
flt = Filter(id="sidebar")
flt.multiselect("Region", options=["North", "South", "East", "West"])
flt.selectbox("Year", options=[2023, 2024, 2025], filter_type="gte")

# Apply filters (cached automatically)
df_filtered = filter_data(df, flt)

# Render chart
chart = Chart(id="main", data=df_filtered, default_y="Sales", default_x="Month")
chart.update_figure()
chart.show_figure()
```

### Date Grouping

```python
chart = Chart(
    id="timeseries",
    data=df,
    date_col="Order Date",
    default_y="Amount",
    default_x="DateGrouping",
    default_color="Category",
)
chart.get_date_grouping(default="Monthly", default_min_num_periods=12)
chart.update_figure()
chart.show_figure()
```

Users can switch between Daily, Weekly, Bi-Weekly, Monthly, Quarterly, and Yearly aggregation with a single dropdown.

---

## Key Features

- **Multi-perspective exploration** — Bar, line, scatter, donut, sunburst, and map charts from one widget
- **Zero-config interactivity** — Dropdowns for every chart dimension render automatically
- **Date intelligence** — Automatic date grouping with period labels and range filtering
- **Polars-native** — Built for Polars DataFrames with Pandas compatibility
- **Color consistency** — Define color maps once, applied everywhere
- **Cached filtering** — `filter_data()` uses Streamlit's caching for instant reruns
- **SQL generation** — `Filter.filter_sql()` produces WHERE clauses for database queries
- **Grouped+Stacked bars** — A unique bar mode that combines grouping and stacking
- **Legend sorting** — Optionally sort legend entries by total value

---

## Documentation

Full API documentation is available at the [Chartly docs site](https://gabenespoli.github.io/chartly/).

---

## Development

```bash
uv venv
uv pip install -e ".[dev]"
pytest
pre-commit run --all-files   # black + isort
```

---

## License

MIT
