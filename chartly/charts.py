from datetime import date

import pandas as pd
import polars as pl
import streamlit as st
from dateutil.relativedelta import relativedelta
from polars import col

from chartly import graphs

BARMODES = {
    "grouped": "group",
    "stacked": "stack",
    "overlaid": "relative",
}


class Chart:
    def __init__(
        self,
        id: str,
        title: str = None,
        data: pl.DataFrame = None,
        y_opts: list = None,
        x_opts: list = None,
        color_opts: list = None,  # also for size, facet_col, facet_row
        size_opts: list = None,
        default_y: str = None,
        default_x: str = None,
        default_color: str = None,
        colormaps: dict = None,
        map_hover_cols: list = None,
        map_hover_name: str = None,
    ):
        self.id = id
        self.title = title or id
        self.data = data
        self.y_opts = y_opts or data.columns
        self.x_opts = x_opts or data.columns
        self.color_opts = color_opts or data.columns
        self.size_opts = size_opts or data.columns
        if None not in self.color_opts:
            self.color_opts = [None] + self.color_opts

        self.default_y = default_y or self.y_opts[0]
        self.default_x = default_x or self.x_opts[0]
        self.default_color = default_color or None

        self.colormaps = colormaps
        self.map_hover_cols = map_hover_cols
        self.map_hover_name = map_hover_name

        self.graph_types = ["bar", "line", "scatter", "donut", "sunburst"]
        if "lat" in data.columns and "lon" in data.columns:
            self.graph_types.append("map")

        self.get_options()

        self.fig = None

    @staticmethod
    def header(title: str):
        cc = st.columns([8, 4, 4, 4, 3])
        cc[0].markdown(f"<h1>| {title}</h1>", unsafe_allow_html=True)
        return cc

    @staticmethod
    def _popover_chart_options_style():
        return '<div style="height: 28px;"></div>'

    def get_options(self):
        cc = self.header(self.title)
        cc[4].write(self._popover_chart_options_style(), unsafe_allow_html=True)
        pp = cc[4].popover("Options")

        self.graph_type = pp.selectbox(
            label="Graph Type",
            options=self.graph_types,
            key=f"{self.id}_graph_type",
        )

        self.y = cc[1].selectbox(
            label="y",
            options=self.y_opts,
            index=self.y_opts.index(self.default_y),
            key=f"{self.id}_y",
        )
        self.x = cc[2].selectbox(
            label="x",
            options=self.x_opts,
            index=self.x_opts.index(self.default_x),
            key=f"{self.id}_x",
        )
        self.color = cc[3].selectbox(
            label="Color",
            options=self.color_opts,
            index=self.color_opts.index(self.default_color),
            key=f"{self.id}_color",
            disabled=self.graph_type == "donut",
        )
        self.facet_col = pp.selectbox(
            label="Column Split",
            options=[None] + self.color_opts,
            key=f"{self.id}_facet_col",
        )
        self.facet_row = pp.selectbox(
            label="Row Split",
            options=[None] + self.color_opts,
            key=f"{self.id}_facet_row",
        )
        self.size = pp.selectbox(
            label="Size",
            options=[None] + self.size_opts,
            key=f"{self.id}_size",
            disabled=self.graph_type not in ["scatter", "map"],
        )

        self.barmode = pp.selectbox(
            label="Bar Mode",
            options=BARMODES.keys(),
            index=1,
            key=f"{self.id}_barmode",
            disabled=self.graph_type != "bar",
        )
        self.marginal = pp.selectbox(
            label="Marginal Plots",
            options=[None, "box", "histogram", "rug", "violin"],
            key=f"{self.id}_marginal",
            disabled=self.graph_type != "scatter",
        )
        self.histogram_bins = pp.number_input(
            label="Marginal Histogram Bins",
            value=50,
            step=5,
            key=f"{self.id}_histogram_bins",
        )

        self.height = pp.number_input(
            label="Height",
            value=600,
            min_value=100,
            max_value=2000,
            step=25,
            key=f"{self.id}_height",
        )

        self.orientation_h = pp.checkbox(
            label="Horizontal bars",
            key=f"{self.id}_orientation",
            disabled=self.graph_type != "bar" or self.x in ["Date", "Month"],
        )
        self.orientation = "h" if self.orientation_h else "v"
        self.sort_legend_by_value = pp.checkbox(
            label="Sort legend by value",
            disabled=True if self.facet_col or self.facet_row else False,
            key=f"{self.id}_sort_legend_by_value",
        )

    def update_figure(
        self,
        orientation: str = None,
        colormaps: dict = None,
        map_theme: str = None,  # Light or Dark
        **kwargs,
    ):
        if self.graph_type == "map":
            if "lat" not in self.data.columns or "lon" not in self.data.columns:
                st.error("Map requires lat and lon columns")
                return

            data_nomap = self.data.filter(
                (col("lat").is_null()) | (col("lon").is_null())
            )
            data_map = self.data.filter(
                ~(col("lat").is_null()) & ~(col("lon").is_null())
            )
            if self.size:
                data_nosize = data_map.filter(
                    (col(self.size).is_null()) | (col(self.size) <= 0)
                )
                data_map = data_map.filter(
                    ~(col(self.size).is_null()) & ~(col(self.size) <= 0)
                )

            self.fig = graphs.map(
                data_map,
                size_col=self.size,
                color_col=self.color,
                legend_hide_title=True,
                colormaps=colormaps or self.colormaps,
                map_theme=map_theme,
                hover_cols=self.map_hover_cols,
                hover_name=self.map_hover_name,
                **kwargs,
            )

        elif self.graph_type == "donut":
            self.fig = graphs.donut(
                self.data,
                values=self.y,
                names=self.x,
                facet_col=self.facet_col,
                facet_row=self.facet_row,
                legend_bottom=True,
                legend_hide_title=True,
                height=self.height,
                colormaps=colormaps or self.colormaps,
            )

        elif self.graph_type == "sunburst":
            path = [
                x for x in [self.x, self.color, self.facet_col, self.facet_row] if x
            ]
            path = list(set(path))
            colormaps = colormaps or self.colormaps
            if colormaps:
                color_discrete_map = colormaps.get(self.color)
            self.fig = graphs.sunburst(
                self.data,
                path=path,
                color=self.color,
                height=self.height,
                color_discrete_map=color_discrete_map,
            )

        else:
            marginal_args = (
                dict(marginal_x=self.marginal, marginal_y=self.marginal)
                if self.marginal and self.graph_type == "scatter"
                else {}
            )

            self.fig = graphs.graph(
                self.data,
                y=self.y,
                x=self.x,
                color=self.color,
                facet_col=self.facet_col,
                facet_row=self.facet_row,
                graph_type=self.graph_type,
                barmode=BARMODES.get(self.barmode, "stack"),
                orientation=orientation or self.orientation,
                height=self.height,
                sort_legend_by_value=self.sort_legend_by_value,
                colormaps=colormaps or self.colormaps,
                **marginal_args,
                **kwargs,
            )

            if self.graph_type == "scatter" and self.marginal == "histogram":
                self.fig = self.fig.update_traces(
                    nbinsx=self.histogram_bins,
                    nbinsy=self.histogram_bins,
                    selector=dict(type="histogram"),
                )

    def show_figure(self, use_container_width=True):
        if self.fig is not None:
            st.plotly_chart(
                self.fig,
                use_container_width=use_container_width,
                key=f"{self.id}_plotly_chart",
            )
        else:
            st.error("Figure is not updated. Call Chart.update_figure() first.")

    def show_data(self):
        with st.expander(
            f"{self.title} Data ({self.data.to_pandas().shape[0]} records)",
            expanded=False,
        ):
            st.dataframe(self.data)

    def highlight_monthly_regions(
        self,
        grouping: str,
        min_date: date,
        max_date: date,
        min_month_chart: date,
        font_color: str = "white",
        fillcolor: str = "#888888",
    ):
        """
        Alternate background color for each period in the chart.

        Args:
            grouping: Year, Half-Year, Quarter, Fiscal Year, Fisal Half-Year,
                Fiscal Quarter

        Returns:
            Updates self.fig with vrects and annotations for each period.
        """
        if self.graph_type != "bar":
            pass

        # get periods (list of dicts)
        # ---------------------------
        max_date = max_date + relativedelta(day=31)

        if grouping == "Year":
            dates = pd.date_range(
                min_date,
                max_date,
                freq="YE-" + max_date.strftime("%b").upper(),
            )

        elif grouping == "Fiscal Year":
            dates = pd.date_range(min_date, max_date, freq="YS-MAR")

        elif grouping == "Half-Year":
            r = relativedelta(max_date, min_date)
            rm = r.years * 12 + r.months
            rd = rm - (rm % 6)
            dates = pd.date_range(
                max_date - relativedelta(months=rd), max_date, freq="6ME"
            )

        elif grouping == "Fiscal Half-Year":
            dates = pd.date_range(min_date, max_date, freq="QS-MAR")
            dates = [dates[0] - relativedelta(months=3)] + list(dates)
            dates = [x for x in dates if x.month in [3, 9]]

        elif grouping == "Quarter":
            dates = pd.date_range(
                min_date,
                max_date,
                freq="QE-" + max_date.strftime("%b").upper(),
            )

        elif grouping == "Fiscal Quarter":
            dates = pd.date_range(min_date, max_date, freq="QS-MAR")
            dates = [dates[0] - relativedelta(months=3)] + list(dates)

        else:
            return []

        dates = [x.date() for x in dates]
        add_final_period = dates[-1] + relativedelta(day=31) < max_date
        dates = [x - relativedelta(day=15) for x in dates]

        periods = []
        for idx in range(len(dates) - 1):
            periods.append(
                dict(
                    x0=dates[idx],
                    x1=dates[idx + 1],
                    fillcolor=fillcolor if idx % 2 == 0 else None,
                    title="",
                    text="",
                )
            )

        if add_final_period:
            periods.append(
                dict(
                    x0=dates[-1],
                    x1=max_date - relativedelta(day=15),
                    fillcolor=fillcolor if len(periods) % 2 == 0 else None,
                    title="",
                    text="<br>*INCOMPLETE PERIOD*",
                )
            )

        # loop periods and add vrects/text
        # --------------------------------
        for idx, period in enumerate(periods):
            fc = font_color if idx % 2 == 0 else None
            first_idx = 0 if idx == 0 else first_idx

            if (period["x0"] < min_month_chart) & (period["x1"] <= min_month_chart):
                # no data during this period
                first_idx = idx + 1
                continue

            elif (period["x0"] < min_month_chart) & (period["x1"] > min_month_chart):
                # partial data during this period
                period["x0"] = min_month_chart + relativedelta(day=15)
                first_idx = idx

            self.fig.add_vrect(
                **{k: v for k, v in period.items() if k in ["x0", "x1", "fillcolor"]},
                layer="below",
                line_width=0,
            )

            if idx > first_idx:
                val0 = val

            period[
                "title"
            ] = f'{period["x0"] + relativedelta(months=1, day=1)} – {period["x1"] + relativedelta(day=31)}'

            val = (
                self.data.filter(
                    (col("Month") >= period["x0"] + relativedelta(months=1, day=1))
                    & (col("Month") <= period["x1"] + relativedelta(day=31))
                )
                .select(pl.sum(self.y))
                .item()
            )

            max_val = (
                self.data.group_by("Month")
                .agg(pl.sum(self.y))
                .select(self.y)
                .max()
                .item()
            )

            if period["title"] != "":
                self.fig.add_annotation(
                    x=period["x0"] + (period["x1"] - period["x0"]) / 2,
                    y=max_val * 1.2,
                    text=period["title"],
                    showarrow=False,
                    font=dict(size=16),
                    font_color=fc,
                )

            self.fig.add_annotation(
                x=period["x0"] + (period["x1"] - period["x0"]) / 2,
                y=max_val * 1.15,
                text=f"{val:,}",
                showarrow=False,
                font=dict(size=16),
                font_color=fc,
            )

            if idx > first_idx:
                valdiff = val - val0
                valdiffpct = valdiff / val0 if val0 != 0 else 1
                text = f"{val - val0:+,} | {valdiffpct:+.1%}"
                if period["text"] != "":
                    diff_font_color = "blue"
                else:
                    diff_font_color = "green" if valdiff > 0 else "red"
                self.fig.add_annotation(
                    x=period["x0"] + (period["x1"] - period["x0"]) / 2,
                    y=max_val * 1.1,
                    text=text,
                    showarrow=False,
                    font=dict(size=16),
                    font_color=diff_font_color,
                )

            if period["text"] != "":
                self.fig.add_annotation(
                    x=period["x0"] + (period["x1"] - period["x0"]) / 2,
                    y=max_val * 1.05,
                    text=period["text"],
                    showarrow=False,
                    font=dict(size=16),
                    font_color=fc,
                )
