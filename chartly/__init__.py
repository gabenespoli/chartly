"""Chartly: Interactive multi-perspective chart exploration for Streamlit.

Chartly wraps Plotly and Streamlit to let you hand it a DataFrame, choose
your fields, set sensible defaults, and instantly explore your data from
many angles — bar, line, scatter, donut, sunburst, map, sankey, and
waterfall — all in one configurable chart widget.

.. include:: ../README.md
"""

from chartly.charts import Chart
from chartly.filter import Filter
from chartly.filter import filter_data
from chartly import filter
from chartly import graphs

__all__ = [
    "Chart",
    "Filter",
    "filter",
    "filter_data",
    "graphs",
]
