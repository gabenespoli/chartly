import warnings
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

from chartly import utils

FONT_SIZE = 16


def millify(
    n: Union[int, float],
    precision: int = 2,
    drop_nulls: bool = True,
    prefixes: Optional[List[str]] = None,
) -> str:
    prefixes = prefixes or []
    # https://github.com/azaitsev/millify
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
    df: pl.DataFrame,
    plot_vars: List[str],
    kwargs: Dict[str, Any],
    colormaps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Set category orders based on the order in the color map.
    Otherwise sort the values alphabetically.

    Returns a new dict; the input kwargs is not mutated.
    """
    colormaps = colormaps or {}
    existing_orders = kwargs.get("category_orders") or {}
    category_orders = {}
    for plot_var in plot_vars:
        col_name = kwargs.get(plot_var)
        if col_name is not None:
            if col_name in existing_orders:
                category_orders[col_name] = existing_orders[col_name]
                continue
            colormap = colormaps.get(col_name)
            if colormap:
                category_orders[col_name] = [
                    x for x in colormap.keys() if x in df[col_name].unique()
                ]
            else:
                try:
                    category_orders[col_name] = df[col_name].unique().sort()
                except TypeError:
                    warnings.warn(
                        f"Column {col_name!r} contains unsortable types (e.g., mixed"
                        " types); skipping category ordering"
                    )
    return {**kwargs, "category_orders": category_orders}


def _get_height(df: pl.DataFrame, kwargs: Dict[str, Any]) -> int:
    """Adjust graph height based on the number of categories that will be plotted with
    facet_row"""
    default_height = 550
    if "height" in kwargs.keys() or kwargs.get("facet_row") is None:
        return default_height
    nunique = df[kwargs.get("facet_row")].n_unique()
    return 800 if nunique > 4 else default_height


def _grouped_stacked_bar(
    df: pl.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    value_col: Optional[str],
    stack_col: str,
    bar_group_col: str,
    orientation: str,
    height: int,
    colormaps: Dict[str, Any],
    sort_legend_by_value: bool,
    text_auto: Optional[Union[str, bool]],
) -> go.Figure:
    """Grouped + Stacked bars: go.Bar traces offset per bar_group, stacked by
    stack_col.

    Note: more than four parameters because this mirrors the resolved plot spec
    of graph(); revisit if this signature keeps growing.
    """
    df = df.sort(x_col)

    fig = go.Figure()

    # Assign consistent colors per stack value
    stack_values = sorted(df[stack_col].unique().to_list())
    colors_palette = px.colors.qualitative.Plotly
    color_map = colormaps.get(stack_col, {}) if stack_col else {}
    if not color_map:
        color_map = {
            val: colors_palette[i % len(colors_palette)]
            for i, val in enumerate(stack_values)
        }

    # Totals per (bar_group, stack) pair; when both come from the same column
    # the pair is (value, value). Tertiary sort keys reproduce pandas' sorted
    # groupby order so trace ordering is unchanged.
    if bar_group_col == stack_col:
        totals = (
            df.group_by(stack_col)
            .agg(col(value_col).sum())
            .rename({stack_col: "_group"})
            .with_columns(col("_group").alias("_stack"))
            .select(["_group", "_stack", value_col])
        )
    else:
        totals = (
            df.group_by([bar_group_col, stack_col])
            .agg(col(value_col).sum())
            .rename({bar_group_col: "_group", stack_col: "_stack"})
        )
    rows = totals.rows(named=True)

    if sort_legend_by_value:
        if bar_group_col == stack_col:
            # When same column, just sort by value descending
            ordered = sorted(rows, key=lambda r: (-float(r[value_col]), r["_stack"]))
            sorted_combinations = [(r["_stack"], r["_stack"]) for r in ordered]
        else:
            # Sort by bar_group first (ascending), then by value descending
            ordered = sorted(
                rows,
                key=lambda r: (r["_group"], -float(r[value_col]), r["_stack"]),
            )
            sorted_combinations = [(r["_group"], r["_stack"]) for r in ordered]

        # Create legend names with values
        legend_name_map = {
            (r["_group"], r["_stack"]): f"{r['_stack']} ({millify(r[value_col])})"
            for r in rows
        }
    else:
        # Default: sort by stack_col only
        ordered = sorted(rows, key=lambda r: (r["_stack"], r["_group"]))
        sorted_combinations = [(r["_group"], r["_stack"]) for r in ordered]
        legend_name_map = {(r["_group"], r["_stack"]): str(r["_stack"]) for r in rows}

    shown_in_legend = set()

    # Create numeric positions
    unique_x = sorted(df[x_col].unique().to_list())
    unique_bar_groups = sorted(df[bar_group_col].unique().to_list())
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
        group_df = df.filter(
            (col(bar_group_col) == grp_val) & (col(stack_col) == stack_val)
        )
        name = legend_name_map[(grp_val, stack_val)]
        show_legend = stack_val not in shown_in_legend
        if show_legend:
            shown_in_legend.add(stack_val)

        x_vals_raw = list(group_df[x_col])
        y_vals = list(group_df[y_col])
        x_vals = [x_pos_map[x] + bar_group_offsets[grp_val] for x in x_vals_raw]

        if orientation == "h":
            fig.add_trace(
                go.Bar(
                    y=x_vals_raw,
                    x=y_vals,
                    name=name,
                    legendgroup=str(stack_val),
                    showlegend=show_legend,
                    marker_color=color_map.get(stack_val),
                    orientation="h",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    name=name,
                    legendgroup=str(stack_val),
                    showlegend=show_legend,
                    marker_color=color_map.get(stack_val),
                )
            )

    # Calculate total for each (x, bar_group) and add labels at top of each bar
    if text_auto:
        totals = df.group_by([x_col, bar_group_col]).agg(col(value_col).sum())
        for x_val, grp_val, total in totals.iter_rows():
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
    return fig


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
    """
    Args:
        legend_reversed: Reverse the order of the legend so it matches the order of the
            colors on the bars.
        legend_hide_title: Hide the title of the legend.
        color_matches_xy: Set the color of the bars to match the x or y axis (depends on
            orientation).
        sort_legend_by_value: Sort the legend by the sum of the values in each category.
        pre_agg_for_text_auto: Pre-aggregate the data to get the proper text_auto
            values.

    Note: the boolean flags here (legend_reversed, legend_hide_title,
    color_matches_xy, sort_legend_by_value, pre_agg_for_text_auto) would normally
    be split into separate functions, but they are kept as flags for API
    compatibility with Chart.update_figure's options popover. Revisit only if a
    major version changes this public surface.
    """
    df = utils.ensure_polars(df)
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
        if agg_func == "mean":
            df = df.group_by(groupby).agg(col(value_col).mean().alias(value_col))
        else:
            df = df.group_by(groupby).agg(col(value_col).sum().alias(value_col))

    kwargs["text_auto"] = True if text_auto is None else text_auto

    if (
        sort_legend_by_value
        and color_col
        and not kwargs.get("bar_group")
        and kwargs.get("facet_col") is None
        and kwargs.get("facet_row") is None
    ):
        color_col_order = dict(
            df.group_by(color_col)
            .agg(col(value_col).sum().alias(value_col))
            .sort(value_col, descending=True)
            .iter_rows()
        )
        color_col_order = {k: f"{k} ({millify(v)})" for k, v in color_col_order.items()}
        df = df.with_columns(col(color_col).replace(color_col_order).alias(color_col))
        kwargs["category_orders"][color_col] = color_col_order.values()
        color_discrete_map = dict()
        for k, v in color_col_order.items():
            color_discrete_map[v] = kwargs.get("color_discrete_map").get(k)
        kwargs["color_discrete_map"] = color_discrete_map

    if graph_type in ["line", "scatter"]:
        df = df.sort(by=[group_col, x_col])
        fig = px.scatter(
            df,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["barmode", "text_auto", "bar_group"]
            },
        )
    elif kwargs.get("bar_group") and color_col:
        # Grouped + Stacked: use go.Bar with offsetgroup for grouping and barmode=stack
        fig = _grouped_stacked_bar(
            df=df,
            x_col=x_col,
            y_col=y_col,
            value_col=value_col,
            stack_col=color_col,
            bar_group_col=kwargs.pop("bar_group"),
            orientation=orientation if orientation else "v",
            height=kwargs.get("height", 550),
            colormaps=colormaps,
            sort_legend_by_value=sort_legend_by_value,
            text_auto=text_auto,
        )
    else:
        # Remove bar_group from kwargs before passing to px.bar
        kwargs.pop("bar_group", None)
        fig = px.bar(df, **kwargs)
    if graph_type == "line":
        fig.update_traces(dict(mode="lines+markers"))

    fig.update_layout(
        font=dict(size=font_size),
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
    """
    - automatically looks for the names, facet_col, and facet_row args, and uses the
    colors module to set color maps and category orders
    """
    df = kwargs.pop("data_frame", None)
    if df is None:
        df = args[0] if args else None
        args = args[1:]
    df = utils.ensure_polars(df)

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
        df,
        *args,
        hole=hole,
        **kwargs,
    )

    fig.update_layout(height=_get_height(df, kwargs))

    fig.update_traces(
        texttemplate="%{percent:.0%} (%{value})",
        textposition="inside",
        textfont=dict(size=font_size),
        sort=sort,  # True to sort by size, False to sort as in df
        direction="clockwise",
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
    geo_infos = {
        "CA": dict(
            scope="north america",
            center={"lat": 60, "lon": -98},
            zoom=2,
            resolution=50,
        ),
        "DE": dict(
            scope="europe",
            center={"lat": 51.5, "lon": 10},
            zoom=4,
            resolution=50,
        ),
        "FR": dict(
            scope="europe",
            center={"lat": 47.5, "lon": 1},
            zoom=4,
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
            resolution=50,
        ),
        "IE": dict(
            scope="europe",
            center={"lat": 54.5, "lon": -3},
            zoom=4,
            resolution=50,
        ),
        "UK/IE": dict(
            scope="europe",
            center={"lat": 54.5, "lon": -3},
            zoom=4,
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
            resolution=50,
            showcountries=True,
        ),
    }
    if country is not None and country not in geo_infos:
        raise ValueError(
            f"Unknown country {country!r}; expected one of {sorted(geo_infos)}"
        )
    return geo_infos.get(country, geo_infos["WORLD"])


# Shadows the builtin within this module deliberately: graphs.map is the
# established public name used by Chart.update_figure and external callers.
# Renaming would break them; revisit only at a major version.
def map(
    df: Union[pd.DataFrame, pl.DataFrame],
    country: Optional[str] = None,
    size_col: Optional[str] = None,
    color_col: Optional[str] = None,
    map_theme: str = "Light",
    hover_cols: Optional[List[str]] = None,
    hover_name: Optional[str] = None,
    legend_hide_title: bool = False,
    lat_col: str = "lat",  # BillingLatitude
    lon_col: str = "lon",  # BillingLongitude
    font_size: int = FONT_SIZE,
    colormaps: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> go.Figure:
    df = utils.ensure_polars(df)
    hover_cols = hover_cols or []
    colormaps = colormaps or {}
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

    map_style = "carto-darkmatter" if map_theme == "Dark" else "open-street-map"
    scatter_kwargs = dict(
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
        height=500,
    )
    # plotly >= 6 replaced Mapbox with MapLibre and renamed the factory plus
    # its style argument
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(df, map_style=map_style, **scatter_kwargs)
    else:
        fig = px.scatter_mapbox(df, mapbox_style=map_style, **scatter_kwargs)

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
    df: Union[pd.DataFrame, pl.DataFrame],
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
    df = utils.ensure_polars(df)

    # Get nodes; maintain_order keeps pandas' first-appearance node order
    labels = {
        node0: len(df),
        **{
            x: int(df[node1].eq(x).sum())
            for x in df[node1].unique(maintain_order=True).to_list()
        },
        **{
            x: int(df[node2].eq(x).sum())
            for x in df[node2].unique(maintain_order=True).to_list()
        },
    }
    node_list = list(labels.keys())

    colors: Optional[Dict[str, str]] = None
    if cmap is not None:
        colors = {k: v for k, v in cmap.items() if k in labels}

    # Define links between nodes
    rows = []
    for n1 in df[node1].unique(maintain_order=True).to_list():
        df_n1 = df.filter(col(node1) == n1)
        rows.append({"source": node0, "target": n1, "value": len(df_n1)})
        for n2 in df[node2].unique(maintain_order=True).to_list():
            rows.append(
                {
                    "source": n1,
                    "target": n2,
                    "value": len(df_n1.filter(col(node2) == n2)),
                }
            )
    links = [r for r in rows if r["value"] != 0]

    # Draw sankey figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=[f"{name} ({count:,})" for name, count in labels.items()],
                    color=(
                        [colors.get(name) for name in node_list]
                        if colors is not None
                        else None
                    ),
                ),
                link=dict(
                    source=[node_list.index(r["source"]) for r in links],
                    target=[node_list.index(r["target"]) for r in links],
                    value=[r["value"] for r in links],
                    color="gray",
                ),
            )
        ]
    )
    fig.update_layout(font=dict(size=22), hovermode=False)
    return fig


def sunburst(df: Union[pd.DataFrame, pl.DataFrame], **kwargs: Any) -> go.Figure:
    # Pure passthrough of px.sunburst. Kept deliberately as a stable seam so
    # callers can treat graph/donut/sunburst uniformly; revisit if the package
    # ever cuts a major version.
    fig = px.sunburst(utils.ensure_polars(df), **kwargs)
    return fig


def waterfall(
    shap_values: Union[pd.DataFrame, pl.DataFrame], n_top_features: int = 9
) -> go.Figure:
    """Waterfall plot for shap values.

    The frame must hold a single row whose columns are the per-feature SHAP
    contributions plus `E[f(x)]` and `f(x)`.
    """
    shap_values = utils.ensure_polars(shap_values)
    base_value = shap_values["E[f(x)]"][0]
    contributions = shap_values.select(pl.exclude(["E[f(x)]", "f(x)"])).row(
        0, named=True
    )
    # Sort ascending by |contribution|; the tail becomes the top features and
    # everything before it is summed into the "other" bucket
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]))
    feature_count = len(ordered)
    n_top_features = min(n_top_features, feature_count)
    features: List[str] = []
    values: List[float] = []
    if feature_count - n_top_features > 0:
        other_sum = sum(v for _, v in ordered[: feature_count - n_top_features])
        features.append(f"Sum of {feature_count - n_top_features} other features")
        values.append(other_sum)
    for name, value in ordered[feature_count - n_top_features :]:
        features.append(name)
        values.append(value)
    fig = go.Figure(
        go.Waterfall(
            name="waterfall",
            base=base_value,
            orientation="h",
            y=features,
            x=values,
            textposition="outside",
            text=["{:+}".format(round(x, 3)) for x in values],
        )
    )
    return fig
