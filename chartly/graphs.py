"""Plotly figure factory functions for various chart types.

This module provides opinionated wrappers around Plotly Express and
Plotly Graph Objects to produce bar, line, scatter, donut, sunburst,
map, sankey, and waterfall charts with automatic color mapping, category
ordering, and legend formatting.
"""

from decimal import Decimal
from math import floor
from math import log10
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from polars import col

FONT_SIZE = 16


def millify(
    n: Union[int, float],
    precision: int = 2,
    drop_nulls: bool = True,
    prefixes: Optional[List[str]] = None,
) -> str:
    """Format a number with human-readable suffixes (k, M, B, T, etc.).

    Based on `millify <https://github.com/azaitsev/millify>`_ by Alexander Zaitsev.

    Args:
        n: The number to format.
        precision: Number of significant digits to keep.
        drop_nulls: If True, remove trailing zeros from the result.
        prefixes: Custom suffix list. Defaults to ["", "k", "M", "B", "T", ...].

    Returns:
        A compact string representation (e.g., "1.5M", "300k").

    Example:
        >>> millify(1_500_000)
        '1.5M'
        >>> millify(42)
        '42'
    """
    prefixes = prefixes or []
    millnames = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    if prefixes:
        millnames = [""]
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(
        0,
        min(len(millnames) - 1, int(floor(0 if n == 0 else log10(abs(n)) / 3))),
    )
    result = "{:.{precision}g}".format(n / 10 ** (3 * millidx), precision=precision)
    if drop_nulls:
        result = Decimal(result)
        result = (
            result.quantize(Decimal(1))
            if result == result.to_integral()
            else result.normalize()
        )
    return "{0}{dx}".format(result, dx=millnames[millidx])


def _add_category_orders(
    df: Union[pd.DataFrame, pl.DataFrame],
    plot_vars: List[str],
    kwargs: Dict[str, Any],
    colormaps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Set category orders for Plotly figures based on color maps or alphabetical sort.

    If a color map is provided for a column, its key order determines the
    category order. Otherwise, unique values are sorted alphabetically.

    Args:
        df: The DataFrame being plotted.
        plot_vars: List of kwargs keys to check (e.g., ["x", "y", "color"]).
        kwargs: The keyword arguments dict being built for Plotly.
        colormaps: Mapping of column names to color dictionaries.

    Returns:
        The updated kwargs dict with ``category_orders`` set.
    """
    category_orders = {}
    colormaps = colormaps or {}
    for plot_var in plot_vars:
        col_name = kwargs.get(plot_var)
        if col_name is not None:
            colormap = colormaps.get(col_name)
            if (
                "category_orders" in kwargs.keys()
                and col_name in kwargs["category_orders"]
            ):
                category_orders[col_name] = kwargs["category_orders"][col_name]
            elif colormap is not None and colormap != {}:
                category_orders[col_name] = [
                    x for x in colormap.keys() if x in df[col_name].unique()
                ]
            else:
                try:
                    category_orders[col_name] = df[col_name].unique().sort()
                except TypeError:
                    print("Column contains unsortable types (e.g., mixed types); skip ordering")
                    pass
    kwargs["category_orders"] = category_orders
    return kwargs


def _get_height(df: Union[pd.DataFrame, pl.DataFrame], kwargs: Dict[str, Any]) -> int:
    """Calculate appropriate chart height based on facet row count.

    Returns a taller chart (800px) when there are more than 4 facet rows,
    otherwise returns the default height (550px).

    Args:
        df: The DataFrame being plotted.
        kwargs: Plot keyword arguments (checked for "height" and "facet_row").

    Returns:
        Chart height in pixels.
    """
    default_height = 550
    if "height" in kwargs.keys() or kwargs.get("facet_row") is None:
        return default_height
    if isinstance(df, pd.DataFrame):
        nunique = df[kwargs.get("facet_row")].nunique()
    elif isinstance(df, pl.DataFrame):
        nunique = df[kwargs.get("facet_row")].n_unique()
    return 800 if nunique > 4 else default_height


def graph(
    df: Union[pd.DataFrame, pl.DataFrame],
    legend_reversed: bool = False,
    legend_hide_title: bool = True,
    font_size: int = FONT_SIZE,
    text_auto: Optional[Union[str, bool]] = None,
    color_matches_xy: bool = False,
    sort_legend_by_value: bool = False,
    pre_agg_for_text_auto: bool = True,
    agg_func: str = "sum",
    graph_type: str = "bar",
    colormaps: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a bar, line, or scatter Plotly figure with automatic aggregation.

    This is the primary graph factory. It pre-aggregates data, applies color
    maps and category ordering, handles orientation flipping, and supports
    grouped+stacked bar charts.

    Args:
        df: Source DataFrame (Pandas or Polars).
        legend_reversed: Reverse legend order to match stacked bar order.
        legend_hide_title: Hide the legend title.
        font_size: Base font size for the figure.
        text_auto: Text label format for bars. True for automatic formatting.
        color_matches_xy: If True and no color column is set, use the x/y
            grouping column as the color.
        sort_legend_by_value: Sort legend entries by their total value
            (descending), appending the total to each legend label.
        pre_agg_for_text_auto: Pre-aggregate data so text labels show correct
            totals per bar segment.
        agg_func: Aggregation function — "sum" or "mean".
        graph_type: One of "bar", "line", or "scatter".
        colormaps: Dictionary mapping column names to color maps.
        **kwargs: Additional Plotly Express arguments (x, y, color, facet_col,
            facet_row, barmode, orientation, height, bar_group, etc.).

    Returns:
        A configured Plotly Figure object.
    """
    colormaps = colormaps or {}

    group_col = kwargs.get("x")
    value_col = kwargs.get("y")
    orientation = kwargs.get("orientation") or "v"
    if orientation == "h":
        x_col = kwargs.get("y")
        y_col = kwargs.get("x")
    else:
        x_col = kwargs.get("x")
        y_col = kwargs.get("y")
    kwargs["x"] = x_col
    kwargs["y"] = y_col

    # set color map
    if kwargs.get("color") is not None:
        color_col = kwargs.get("color")
        kwargs["color"] = color_col
        kwargs["color_discrete_map"] = colormaps.get(color_col)
    elif color_matches_xy:
        kwargs["color"] = group_col
        kwargs["color_discrete_map"] = colormaps.get(group_col)
        color_col = group_col
    else:
        color_col = None

    kwargs = _add_category_orders(
        df,
        plot_vars=["x", "y", "color", "facet_col", "facet_row"],
        kwargs=kwargs,
        colormaps=colormaps,
    )

    if "height" not in kwargs:
        kwargs["height"] = _get_height(df, kwargs)

    if pre_agg_for_text_auto:
        groupby = [
            x
            for x in [
                kwargs.get("color"),
                kwargs.get("facet_col"),
                kwargs.get("facet_row"),
                kwargs.get("bar_group"),
            ]
            if x is not None
        ]
        groupby = list(set([group_col] + groupby))
        if isinstance(df, pd.DataFrame):
            if agg_func == "mean":
                df = df.groupby(groupby)[value_col].mean().reset_index()
            else:
                df = df.groupby(groupby)[value_col].sum().reset_index()
        elif isinstance(df, pl.DataFrame):
            if agg_func == "mean":
                df = df.group_by(groupby).agg(col(value_col).mean().alias(value_col))
            else:
                df = df.group_by(groupby).agg(col(value_col).sum().alias(value_col))
        else:
            raise ValueError("df should be a pandas or polars DataFrame")

    text_auto = text_auto or True
    kwargs["text_auto"] = text_auto

    if (
        sort_legend_by_value
        and color_col
        and not kwargs.get("bar_group")
        and kwargs.get("facet_col") is None
        and kwargs.get("facet_row") is None
    ):
        if isinstance(df, pd.DataFrame):
            color_col_order = (
                df.groupby(color_col)[value_col]
                .sum()
                .sort_values(ascending=False)
                .to_dict()
            )
            color_col_order = {
                k: f"{k} ({millify(v)})" for k, v in color_col_order.items()
            }
            df[color_col] = df[color_col].map(color_col_order)
        elif isinstance(df, pl.DataFrame):
            color_col_order = dict(
                df.group_by(color_col)
                .agg(col(value_col).sum().alias(value_col))
                .sort(value_col, descending=True)
                .iter_rows()
            )
            color_col_order = {
                k: f"{k} ({millify(v)})" for k, v in color_col_order.items()
            }
            df = df.with_columns(
                col(color_col).replace(color_col_order).alias(color_col)
            )
        kwargs["category_orders"][color_col] = color_col_order.values()
        color_discrete_map = dict()
        for k, v in color_col_order.items():
            color_discrete_map[v] = kwargs.get("color_discrete_map").get(k)
        kwargs["color_discrete_map"] = color_discrete_map

    if graph_type in ["line", "scatter"]:
        df = df.sort(by=[group_col, x_col])
        fig = px.scatter(
            df, **{k: v for k, v in kwargs.items() if k not in ["barmode", "text_auto", "bar_group"]}
        )
    elif kwargs.get("bar_group") and color_col:
        # Grouped + Stacked: use go.Bar with offsetgroup for grouping and barmode=stack
        bar_group_col = kwargs.pop("bar_group")
        stack_col = color_col
        height = kwargs.get("height", 550)
        orientation = kwargs.get("orientation", "v")

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        df = df.sort_values(by=x_col)

        fig = go.Figure()

        # Assign consistent colors per stack value
        stack_values = sorted(df[stack_col].unique())
        colors_palette = px.colors.qualitative.Plotly
        color_map = colormaps.get(stack_col, {}) if stack_col else {}
        if not color_map:
            color_map = {
                val: colors_palette[i % len(colors_palette)]
                for i, val in enumerate(stack_values)
            }

        # Calculate totals for sorting legend by value
        # Handle case where bar_group_col and stack_col are the same
        if bar_group_col == stack_col:
            stack_totals = df.groupby(stack_col)[value_col].sum()
        else:
            stack_totals = df.groupby([bar_group_col, stack_col])[value_col].sum()

        if sort_legend_by_value:
            if bar_group_col == stack_col:
                # When same column, just sort by value descending
                sorted_combinations = [(val, val) for val in stack_totals.sort_values(ascending=False).index]
            else:
                # Sort by bar_group first (ascending), then by value descending
                df_sort = stack_totals.reset_index()
                df_sort = df_sort.sort_values(by=[bar_group_col, value_col], ascending=[True, False])
                sorted_combinations = list(zip(df_sort[bar_group_col], df_sort[stack_col]))

            # Create legend names with values
            if bar_group_col == stack_col:
                legend_name_map = {
                    (val, val): f"{val} ({millify(tot)})"
                    for val, tot in stack_totals.items()
                }
            else:
                legend_name_map = {
                    (grp_val, stack_val): f"{stack_val} ({millify(tot)})"
                    for (grp_val, stack_val), tot in stack_totals.items()
                }
        else:
            # Default: sort by stack_col only
            if bar_group_col == stack_col:
                sorted_combinations = [(val, val) for val in sorted(stack_totals.index)]
                legend_name_map = {
                    (val, val): str(val)
                    for val in stack_totals.index
                }
            else:
                sorted_combinations = sorted(stack_totals.index.tolist(), key=lambda x: x[1])
                legend_name_map = {
                    (grp_val, stack_val): str(stack_val)
                    for grp_val, stack_val in stack_totals.index
                }

        shown_in_legend = set()
        
        # Create numeric positions
        unique_x = sorted(df[x_col].unique())
        unique_bar_groups = sorted(df[bar_group_col].unique())
        num_bar_groups = len(unique_bar_groups)
        
        # Create position map: {x_val: numeric_position}
        x_pos_map = {x: i for i, x in enumerate(unique_x)}
        
        # Calculate offset per bar_group to place bars side by side
        offset = 0.4 / max(num_bar_groups, 1)
        bar_group_offsets = {
            grp: (i - (num_bar_groups - 1) / 2) * offset * 2
            for i, grp in enumerate(unique_bar_groups)
        }
        
        # Create mapping from (x_val, bar_group) to numeric x position
        x_bar_pos = {}
        for x_val in unique_x:
            base_pos = x_pos_map[x_val]
            for grp in unique_bar_groups:
                x_bar_pos[(x_val, grp)] = base_pos + bar_group_offsets[grp]
        
        for grp_val, stack_val in sorted_combinations:
            group_df = df[(df[bar_group_col] == grp_val) & (df[stack_col] == stack_val)]
            name = legend_name_map[(grp_val, stack_val)]
            show_legend = stack_val not in shown_in_legend
            if show_legend:
                shown_in_legend.add(stack_val)

            x_vals_raw = list(group_df[x_col])
            y_vals = list(group_df[y_col])
            x_vals = [x_pos_map[x] + bar_group_offsets[grp_val] for x in x_vals_raw]

            if orientation == "h":
                fig.add_trace(go.Bar(
                    y=x_vals_raw,
                    x=y_vals,
                    name=name,
                    legendgroup=str(stack_val),
                    showlegend=show_legend,
                    marker_color=color_map.get(stack_val),
                    orientation="h",
                ))
            else:
                fig.add_trace(go.Bar(
                    x=x_vals,
                    y=y_vals,
                    name=name,
                    legendgroup=str(stack_val),
                    showlegend=show_legend,
                    marker_color=color_map.get(stack_val),
                ))

        # Calculate total for each (x, bar_group) and add labels at top of each bar
        if text_auto:
            totals = df.groupby([x_col, bar_group_col])[value_col].sum()
            for (x_val, grp_val), total in totals.items():
                x_pos = x_bar_pos[(x_val, grp_val)]
                if orientation == "h":
                    fig.add_annotation(
                        x=total,
                        y=x_val,
                        text=millify(total),
                        showarrow=False,
                        xanchor="left",
                        xshift=5,
                        yref="y",
                        xref="x",
                    )
                else:
                    fig.add_annotation(
                        x=x_pos,
                        y=total,
                        text=millify(total),
                        showarrow=False,
                        yanchor="bottom",
                        yshift=5,
                        xref="x",
                        yref="y",
                    )

        # Set up x-axis with tick marks at center of each group
        tick_positions = [x_pos_map[x] for x in unique_x]
        fig.update_layout(
            barmode="stack",
            height=height,
            xaxis=dict(
                tickmode="array",
                tickvals=tick_positions,
                ticktext=unique_x,
            ),
        )
    else:
        # Remove bar_group from kwargs before passing to px.bar
        kwargs.pop("bar_group", None)
        fig = px.bar(df, **kwargs)
    if graph_type == "line":
        fig.update_traces(dict(mode="lines+markers"))

    # fig.update_traces(
    #     textposition="inside",
    # )

    fig.update_layout(
        font=dict(size=font_size),
        # uniformtext=dict(minsize=14),
        # paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            font=dict(size=font_size),
            traceorder="reversed" if legend_reversed else None,
            title=kwargs.get("color") if not legend_hide_title else None,
        ),
    )

    if kwargs.get("facet_row") is not None or kwargs.get("facet_col") is not None:
        fig.for_each_annotation(
            lambda x: (
                x.update(text=x.text.split("=")[-1])
                if not any(
                    [
                        x.text.endswith("True"),
                        x.text.endswith("False"),
                        x.text.endswith(".0"),
                    ]
                )
                else x
            )
        )

    return fig


def donut(
    *args: Any,
    legend_reversed: bool = False,
    legend_bottom: bool = False,
    showlegend: bool = True,
    sort: bool = False,
    legend_hide_title: bool = False,
    font_size: int = FONT_SIZE,
    colormaps: Optional[Dict[str, Any]] = None,
    hole: float = 0.35,
    **kwargs: Any,
) -> go.Figure:
    """Create a donut (pie with hole) chart with automatic color mapping.

    Automatically applies color maps and category ordering based on the
    ``names`` column. Displays percentage and value inside each slice.

    Args:
        *args: Positional arguments; the first should be the DataFrame.
        legend_reversed: Reverse legend order.
        legend_bottom: Place legend below the chart horizontally.
        showlegend: Whether to show the legend.
        sort: If True, sort slices by size. If False, maintain DataFrame order.
        legend_hide_title: Hide the legend title.
        font_size: Base font size.
        colormaps: Dictionary mapping column names to color maps.
        hole: Size of the donut hole (0 to 1). Default 0.35.
        **kwargs: Additional Plotly Express pie arguments (values, names,
            facet_col, facet_row, height, etc.).

    Returns:
        A configured Plotly Figure object.
    """
    df = kwargs.get("data_frame")
    df = args[0] if df is None else df

    # set color map
    if "color" not in kwargs:
        color_col = kwargs.get("names")
        kwargs["color"] = color_col
        kwargs["color_discrete_map"] = colormaps.get(color_col)

    kwargs = _add_category_orders(
        df,
        plot_vars=["names", "facet_col", "facet_row"],
        kwargs=kwargs,
        colormaps=colormaps,
    )

    fig = px.pie(
        *args,
        hole=hole,
        **kwargs,
    )

    fig.update_layout(height=_get_height(df, kwargs))

    fig.update_traces(
        # insidetextorientation="radial",
        # textinfo="percent",
        texttemplate="%{percent:.0%} (%{value})",
        textposition="inside",
        textfont=dict(size=font_size),
        # rotation=90,
        sort=sort,  # True to sort by size, False to sort as in df
        direction="clockwise",
        # hovertemplate=None,
        # hoverinfo="skip",
        showlegend=showlegend,
    )

    fig.update_layout(
        font=dict(size=font_size),
        uniformtext=dict(minsize=14),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            font=dict(size=font_size),
            traceorder="reversed" if legend_reversed else None,
            title=kwargs.get("color") if not legend_hide_title else None,
        ),
    )

    if legend_bottom:
        fig.update_layout(
            legend=dict(
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                orientation="h",
            )
        )

    if kwargs.get("facet_col") or kwargs.get("facet_row"):
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))

    return fig


def get_geo_info(country: Optional[str]) -> Dict[str, Any]:
    """Get map center, zoom, and scope configuration for a country code.

    Args:
        country: ISO country code or region identifier (e.g., "US", "CA", "DE",
            "UK", "WORLD"). Returns world config if not recognized.

    Returns:
        Dictionary with keys like ``scope``, ``center``, ``zoom``, and
        ``resolution`` suitable for Plotly mapbox configuration.
    """
    geo_infos = {
        "CA": dict(
            scope="north america",
            center={"lat": 60, "lon": -98},
            zoom=2,
            # lataxis_range=[48, 57],
            # lonaxis_range=[-140, -30],
            resolution=50,
        ),
        "DE": dict(
            scope="europe",
            center={"lat": 51.5, "lon": 10},
            zoom=4,
            # lataxis_range=[48, 56],  # north-south
            # lonaxis_range=[6, 16],  # east-west
            resolution=50,
        ),
        "FR": dict(
            scope="europe",
            center={"lat": 47.5, "lon": 1},
            zoom=4,
            # lataxis_range=[48, 56],  # north-south
            # lonaxis_range=[6, 16],  # east-west
            resolution=50,
        ),
        "US": dict(
            scope="usa",
            center={"lat": 44, "lon": -98},
            zoom=2,
        ),
        "UK": dict(
            scope="europe",
            center={"lat": 54.5, "lon": -3},
            zoom=4,
            # lataxis_range=[49, 61],
            # lonaxis_range=[-12, 3],
            resolution=50,
        ),
        "IE": dict(
            scope="europe",
            center={"lat": 54.5, "lon": -3},
            zoom=4,
            # lataxis_range=[49, 61],
            # lonaxis_range=[-12, 3],
            resolution=50,
        ),
        "UK/IE": dict(
            scope="europe",
            center={"lat": 54.5, "lon": -3},
            zoom=4,
            # lataxis_range=[49, 61],
            # lonaxis_range=[-12, 3],
            resolution=50,
        ),
        "NA": dict(
            scope="north america",
            lataxis_range=[25, 67],
            resolution=50,
        ),
        "WORLD": dict(
            scope="world",
            center={"lat": 43, "lon": -60},
            zoom=1,
            # lataxis_range=[20, 67],
            # lonaxis_range=[-150, 30],
            resolution=50,
            showcountries=True,
        ),
    }
    return geo_infos.get(country, geo_infos["WORLD"])


def map(
    df: pl.DataFrame,
    country: Optional[str] = None,
    size_col: Optional[str] = None,
    color_col: Optional[str] = None,
    map_theme: str = "Light",
    hover_cols: Optional[List[str]] = None,
    hover_name: Optional[str] = None,
    legend_hide_title: bool = False,
    lat_col: str = "lat",
    lon_col: str = "lon",
    font_size: int = FONT_SIZE,
    colormaps: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> go.Figure:
    """Create a scatter mapbox figure from lat/lon data.

    Renders points on an interactive map with optional size and color encoding.

    Args:
        df: Polars DataFrame with latitude and longitude columns.
        country: Country/region code for default map centering (see
            :func:`get_geo_info`).
        size_col: Column name for bubble size. Negative values are clamped to 0.
        color_col: Column name for point color encoding.
        map_theme: Map tile style — "Light" (OpenStreetMap) or "Dark"
            (CARTO dark matter).
        hover_cols: Additional columns to show in hover tooltips.
        hover_name: Column whose values label points on hover.
        legend_hide_title: Hide the legend title.
        lat_col: Name of the latitude column. Default "lat".
        lon_col: Name of the longitude column. Default "lon".
        font_size: Base font size for legend.
        colormaps: Dictionary mapping column names to color maps.

    Returns:
        A configured Plotly Figure with mapbox scatter.
    """
    hover_cols = hover_cols or []
    geo_info = get_geo_info(country)

    if lat_col is not None:
        df = df.rename({lat_col: "lat"})
    if lon_col is not None:
        df = df.rename({lon_col: "lon"})
    if size_col is not None:
        df = df.with_columns(
            pl.when(col(size_col) < 0).then(0).otherwise(col(size_col)).alias(size_col)
        )
    if color_col is not None:
        df = df.sort(color_col)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        center=geo_info.get("center"),
        zoom=geo_info.get("zoom"),
        size=size_col,
        color=color_col,
        color_discrete_map=colormaps.get(color_col),
        opacity=1,
        hover_data=hover_cols,
        hover_name=hover_name,
        mapbox_style="carto-darkmatter" if map_theme == "Dark" else "open-street-map",
        height=500,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="all",
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            title=color_col if not legend_hide_title else None,
        ),
    )

    return fig


def sankey(
    df: pd.DataFrame,
    node1: str,
    node2: str,
    node0: str = "Total",
    cmap: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Easily draw a 3-level sankey from dataframe columns.

    By default, the first level of the sankey is all rows. Specify two columns, node1
    and node2, whose values will represent nodes for the next two levels. To customize
    the label of the "Total" node, specify node0.

    """
    # Get nodes
    labels = {
        "Total Accounts": len(df),
        **{x: sum(df[node1] == x) for x in list(df[node1].unique())},
        **{x: sum(df[node2] == x) for x in list(df[node2].unique())},
    }
    nodes = pd.DataFrame(data=labels.values(), index=labels.keys(), columns=["value"])

    if cmap is not None:
        cmap = {k: v for k, v in cmap.items() if k in labels}
        cmap = pd.DataFrame(data=cmap.values(), index=cmap.keys(), columns=["color"])
        nodes = nodes.join(cmap)

    nodes = nodes.reset_index()
    nodes = nodes.rename(columns={"index": "node"})
    node_list = nodes["node"].to_list()

    def add_link(
        links: pd.DataFrame,
        source: str,
        target: str,
        value: int,
    ) -> pd.DataFrame:
        row = dict()
        row["source"] = source
        row["target"] = target
        row["value"] = value
        row = pd.DataFrame.from_dict({k: [v] for k, v in row.items()})
        return pd.concat([links, row])

    # Define links between nodes
    links = pd.DataFrame()
    for n1 in df[node1].unique():
        links = add_link(
            links,
            source=node0,
            target=n1,
            value=len(df[df[node1] == n1]),
        )
        for n2 in df[node2].unique():
            links = add_link(
                links,
                source=n1,
                target=n2,
                value=len(df[(df[node1] == n1) & (df[node2] == n2)]),
            )
    links = links[links["value"] != 0]
    links = links.reset_index(drop=True)

    # Draw sankey figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=[f"{x.node} ({x.value:,})" for x in nodes.itertuples()],
                    color=nodes["color"] if cmap is not None else None,
                ),
                link=dict(
                    source=[node_list.index(x) for x in links["source"]],
                    target=[node_list.index(x) for x in links["target"]],
                    value=links["value"],
                    color="gray",
                ),
            )
        ]
    )
    fig.update_layout(font=dict(size=22), hovermode=False)
    return fig


def sunburst(df: Union[pd.DataFrame, pl.DataFrame], **kwargs: Any) -> go.Figure:
    """Create a sunburst chart for hierarchical data exploration.

    Args:
        df: Source DataFrame (Pandas or Polars).
        **kwargs: Arguments passed directly to ``plotly.express.sunburst()``
            (e.g., path, values, color, color_discrete_map, height).

    Returns:
        A Plotly sunburst Figure.
    """
    fig = px.sunburst(df, **kwargs)
    return fig


def waterfall(sv1: pd.DataFrame, n_top_features: int = 9) -> go.Figure:
    """Waterfall plot for shap values.

    The id should be the index of the pandas dataframe
    """
    n_other_features = sv1.shape[1] - n_top_features
    base_value = sv1["E[f(x)]"].iloc[0]
    sv1 = sv1.drop(columns=["E[f(x)]", "f(x)"])
    sv1.index = ["shap_value"]
    sv1 = sv1.T
    # sv1 = sv1.rename(columns={bs: "shap_value"})
    sv1["abs"] = sv1["shap_value"].abs()
    sv1 = sv1.sort_values("abs").drop(columns="abs")
    tmp = sv1.tail(n_top_features).reset_index().rename(columns={"index": "Feature"})
    sv1 = sv1.head(sv1.shape[0] - n_top_features)["shap_value"].sum()
    sv1 = pd.DataFrame(
        {
            "Feature": [f"Sum of {n_other_features} other features"],
            "shap_value": [sv1],
        }
    )
    sv1 = pd.concat([sv1, tmp]).reset_index(drop=True)
    fig = go.Figure(
        go.Waterfall(
            name="waterfall",
            base=base_value,
            orientation="h",
            y=sv1["Feature"],
            x=sv1["shap_value"],
            textposition="outside",
            # text=["+60", "+80", "", "-40", "-20", "Total"],
            text=["{:+}".format(round(x, 3)) for x in sv1["shap_value"]],
            # connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    return fig
