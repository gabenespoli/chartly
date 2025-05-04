from decimal import Decimal
from math import floor
from math import log10

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from polars import col

FONT_SIZE = 16


def millify(n, precision=2, drop_nulls=True, prefixes=[]):
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
    df,
    plot_vars: list,
    kwargs: dict,
    colormaps: dict = {},
) -> dict:
    """Set category orders based on the order in the color map.
    Otherwise sort the values alphabetically.
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
                except Exception:
                    pass
    kwargs["category_orders"] = category_orders
    return kwargs


def _get_height(df, kwargs: dict) -> dict:
    """Adjust graph height based on the number of categories that will be plotted with
    facet_row"""
    default_height = 550
    if "height" in kwargs.keys() or kwargs.get("facet_row") is None:
        return default_height
    if isinstance(df, pd.DataFrame):
        nunique = df[kwargs.get("facet_row")].nunique()
    elif isinstance(df, pl.DataFrame):
        nunique = df[kwargs.get("facet_row")].n_unique()
    return 800 if nunique > 4 else default_height


def graph(
    df,
    legend_reversed: bool = False,
    # legend_bottom: bool = False,
    # showlegend: bool = True,
    # sort: bool = False,
    legend_hide_title: bool = True,
    font_size: int = FONT_SIZE,
    text_auto: str = None,
    color_matches_xy: bool = False,
    sort_legend_by_value: bool = False,
    pre_agg_for_text_auto: bool = True,
    agg_func: str = "sum",
    graph_type: str = "bar",
    colormaps: dict = {},
    **kwargs,
):
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
            df, **{k: v for k, v in kwargs.items() if k not in ["barmode", "text_auto"]}
        )
    else:
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
    *args,
    legend_reversed: bool = False,
    legend_bottom: bool = False,
    showlegend: bool = True,
    sort: bool = False,
    legend_hide_title: bool = False,
    font_size: int = FONT_SIZE,
    colormaps: dict = None,
    hole: float = 0.35,
    **kwargs,
):
    """
    - automatically looks for the names, facet_col, and facet_row args, and uses the
    colors module to set color maps and category orders
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


def get_geo_info(country: str):
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
    df,
    country: str = None,
    size_col: str = None,
    color_col: str = None,
    map_theme: str = "Light",
    hover_cols: list = [],
    hover_name: str = None,
    legend_hide_title: bool = False,
    lat_col: str = "lat",  # BillingLatitude
    lon_col: str = "lon",  # BillingLongitude
    font_size: int = FONT_SIZE,
    colormaps: dict = None,
    **_,
):
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
    df: pd.DataFrame, node1: str, node2: str, node0: str = "Total", cmap: dict = None
):
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


def sunburst(df, **kwargs):
    fig = px.sunburst(df, **kwargs)
    return fig


def waterfall(sv1: pd.DataFrame, n_top_features: int = 9):
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
