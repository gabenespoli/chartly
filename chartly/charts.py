"""High-level Chart component for interactive data exploration in Streamlit.

This module provides the :class:`Chart` class which renders a full interactive
chart widget — including axis selectors, graph-type picker, color/facet/size
options, date grouping, and the Plotly figure itself — with a single
instantiation call.
"""

from datetime import date
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from dateutil.relativedelta import relativedelta
from polars import col

from chartly import graphs
from chartly import utils

BARMODES = {
    "grouped": "group",
    "stacked": "stack",
    "overlaid": "relative",
    "grouped+stacked": "stack",
}

DATE_GROUPING_MAP = {
    "Daily": "1d",
    "Weekly": "1w",
    "Bi-Weekly": "2w",
    "Monthly": "1mo",
    "Quarterly": "1q",
    "Yearly": "1y",
}

# strftime formats for period labels; %G pairs ISO week numbers with their ISO
# year. Bi-Weekly and Quarterly are computed manually (see helpers below).
PERIOD_DATE_FORMATS = {
    "Daily": "%Y-%m-%d",
    "Weekly": "%G-W%V",
    "Monthly": "%Y-%m",
    "Yearly": "%Y",
}

# Columns referenced by name inside aggregations and filters
AMOUNT_COL = "Amount"
MONTH_COL = "Month"
DATE_GROUPING_COL = "DateGrouping"

# Legacy axis column names for which horizontal bars are disabled
HORIZONTAL_BAR_BLOCKED_COLS = ("Date", "Month")


def _polars_date_grouping_column(
    date_col: str, date_grouping: Optional[str]
) -> "pl.Expr":
    """DateGrouping label expression for a polars date/datetime column."""
    if date_grouping == "Bi-Weekly":
        week_num = col(date_col).dt.week()
        year = col(date_col).dt.iso_year().cast(pl.Utf8)
        bi_week = ((week_num - 1) // 2 * 2 + 1).cast(pl.Utf8).str.zfill(2)
        return year + "-W" + bi_week
    if date_grouping == "Quarterly":
        quarter = ((col(date_col).dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        year = col(date_col).dt.year().cast(pl.Utf8)
        return year + "-Q" + quarter
    return col(date_col).dt.strftime(PERIOD_DATE_FORMATS.get(date_grouping, "%Y-%m-%d"))


class Chart:
    """Interactive chart widget that renders Streamlit controls and a Plotly figure.

    Instantiate a ``Chart`` with your DataFrame and field options. The widget
    automatically renders dropdown selectors for graph type, x/y axes, color,
    facets, bar mode, size, marginal plots, and more. Call
    :meth:`update_figure` and :meth:`show_figure` to display the chart.

    Example:
        >>> import polars as pl
        >>> from chartly import Chart
        >>> df = pl.DataFrame({"Category": ["A", "B"], "Sales": [100, 200]})
        >>> chart = Chart(
        ...     id="sales_chart",
        ...     data=df,
        ...     default_x="Category",
        ...     default_y="Sales",
        ... )
        >>> chart.update_figure()
        >>> chart.show_figure()
    """

    def __init__(
        self,
        id: str,
        data: Union[pd.DataFrame, pl.DataFrame],
        title: Optional[str] = None,
        y_opts: Optional[List[str]] = None,
        x_opts: Optional[List[str]] = None,
        color_opts: Optional[
            List[Optional[str]]
        ] = None,  # also for size, facet_col, facet_row
        size_opts: Optional[List[str]] = None,
        default_y: Optional[str] = None,
        default_x: Optional[str] = None,
        default_color: Optional[str] = None,
        default_graph_type: Optional[str] = None,
        default_facet_col: Optional[str] = None,
        default_facet_row: Optional[str] = None,
        default_size: Optional[str] = None,
        default_barmode: str = "stacked",
        default_bar_group: Optional[str] = None,
        default_marginal: Optional[str] = None,
        default_date_grouping: Optional[str] = "Monthly",
        default_height: int = 600,
        default_histogram_bins: int = 50,
        default_orientation_h: bool = False,
        default_sort_legend_by_value: bool = False,
        colormaps: Optional[Dict[str, Any]] = None,
        date_col: Optional[str] = None,
        date_grouping: Optional[str] = None,
        map_hover_cols: Optional[List[str]] = None,
        map_hover_name: Optional[str] = None,
    ) -> None:
        """Initialize the Chart widget and render option controls.

        Args:
            id: Unique identifier for this chart instance (used for Streamlit widget keys).
            title: Display title shown above the chart. Defaults to ``id``.
            data: The source DataFrame (Polars or Pandas) to visualize.
            y_opts: Column names available for the y-axis selector.
                Defaults to all columns in ``data``.
            x_opts: Column names available for the x-axis selector.
                Defaults to all columns in ``data``.
            color_opts: Column names available for the color (and facet/size) selectors.
                ``None`` is prepended automatically to allow "no color" selection.
            size_opts: Column names available for the bubble-size selector.
                Defaults to all columns in ``data``.
            default_y: Default selected y-axis column.
            default_x: Default selected x-axis column.
            default_color: Default selected color column.
            default_graph_type: Default graph type (bar, line, scatter, donut, sunburst, map).
            default_facet_col: Default column for faceted column splits.
            default_facet_row: Default column for faceted row splits.
            default_size: Default column for marker/bubble size.
            default_barmode: Default bar mode — "grouped", "stacked", "overlaid", or
                "grouped+stacked".
            default_bar_group: Default column for grouped+stacked bar grouping.
            default_marginal: Default marginal plot type for scatter (box, histogram,
                rug, violin).
            default_date_grouping: Default date grouping level (Daily, Weekly,
                Bi-Weekly, Monthly, Quarterly, Yearly).
            default_height: Default chart height in pixels.
            default_histogram_bins: Default number of bins for marginal histograms.
            default_orientation_h: If True, default to horizontal bars.
            default_sort_legend_by_value: If True, sort legend entries by total value.
            colormaps: Dictionary mapping column names to color maps
                (e.g., ``{"Status": {"Active": "green", "Inactive": "red"}}``).
            date_col: Name of the datetime column for date grouping.
            date_grouping: Initial date grouping to apply (overrides default_date_grouping
                if provided).
            map_hover_cols: Additional columns to display on map hover tooltips.
            map_hover_name: Column whose values label map markers on hover.

        Note:
            Rendering Streamlit widgets in the constructor mixes object construction
            with UI side effects. Kept deliberately: every existing caller expects
            options to render on instantiation. Revisit if callers are ever migrated
            to an explicit ``chart.render_options()`` call.
        """
        self.id = id
        self.title = title or id
        # pandas input is converted once here; everything downstream is polars
        self.data = utils.ensure_polars(data)
        data = self.data
        self.y_opts = y_opts or list(data.columns)
        self.x_opts = x_opts or list(data.columns)
        self.color_opts = color_opts or list(data.columns)
        if len(self.color_opts) > 0 and None not in self.color_opts:
            self.color_opts = [None] + self.color_opts
        self.size_opts = size_opts or list(data.columns)

        self.graph_types = ["bar", "line", "scatter", "donut", "sunburst"]
        if "lat" in data.columns and "lon" in data.columns:
            self.graph_types.append("map")

        self.colormaps = colormaps
        self.date_col = date_col
        self.date_grouping = date_grouping
        self.min_date_grouping: Optional[str] = None
        self.max_date_grouping: Optional[str] = None
        self.data_chart = self.data

        self.map_hover_cols = map_hover_cols
        self.map_hover_name = map_hover_name
        self.data_nomap = None
        self.data_nosize = None

        self.default_y = default_y or self.y_opts[0]
        self.default_x = default_x or self.x_opts[0]
        self.default_color = default_color
        self.default_graph_type = default_graph_type or self.graph_types[0]
        self.default_facet_col = default_facet_col
        self.default_facet_row = default_facet_row
        self.default_size = default_size
        self.default_barmode = default_barmode
        self.default_bar_group = default_bar_group
        self.default_marginal = default_marginal
        self.default_date_grouping = default_date_grouping or "Monthly"
        self.default_height = default_height
        self.default_histogram_bins = default_histogram_bins
        self.default_orientation_h = default_orientation_h
        self.default_sort_legend_by_value = default_sort_legend_by_value

        self.get_options()

        self.fig = None

    @staticmethod
    def header(title: str) -> List[Any]:
        """Render a Streamlit header row with columns for title and selectors.

        Args:
            title: The title text to display in the header.

        Returns:
            A list of Streamlit column containers for placing additional widgets.
        """
        cols = st.columns([8, 4, 4, 4, 3])
        cols[0].markdown(f"<h1>| {title}</h1>", unsafe_allow_html=True)
        return cols

    @staticmethod
    def _popover_chart_options_style() -> str:
        return '<div style="height: 28px;"></div>'

    @staticmethod
    def group_by_date(
        df: Union[pd.DataFrame, pl.DataFrame],
        date_grouping: Optional[str],
        date_col: str = "Datetime",
        grp_col: Optional[str] = None,
        extra_grp_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Aggregate a DataFrame by a time period, summing an 'Amount' column.

        Groups the data by the specified date granularity and any additional
        grouping columns, then sums the ``Amount`` column. Adds a
        ``DateGrouping`` string column representing the period label.

        Args:
            df: Input DataFrame (Pandas or Polars) containing a date column and
                an ``Amount`` column to aggregate. Converted to Polars internally.
            date_grouping: Grouping level — one of "Daily", "Weekly",
                "Bi-Weekly", "Monthly", "Quarterly", "Yearly", or None (no grouping).
            date_col: Name of the datetime column to group on.
            grp_col: Optional additional column to include in the group-by.
            extra_grp_cols: Additional columns to include in the group-by.

        Returns:
            A new Polars DataFrame aggregated by the specified period with a
            ``DateGrouping`` column added.
        """
        if date_grouping is None:
            return utils.ensure_polars(df)
        df = utils.ensure_polars(df)
        extra_grp_cols = extra_grp_cols or []
        all_grp_cols = [x for x in [grp_col] + extra_grp_cols if x is not None]
        all_grp_cols = list(dict.fromkeys(all_grp_cols))
        df = df.sort(*all_grp_cols, date_col)
        df = df.group_by_dynamic(
            date_col,
            every=DATE_GROUPING_MAP[date_grouping],
            group_by=all_grp_cols if all_grp_cols else None,
        ).agg(col(AMOUNT_COL).sum())
        df = df.with_columns(
            _polars_date_grouping_column(date_col, date_grouping).alias(
                DATE_GROUPING_COL
            )
        )
        return df

    def get_date_grouping(
        self,
        default: Optional[str] = None,
        default_min: Optional[str] = None,
        default_max: Optional[str] = None,
        default_max_complete_period: Optional[date] = None,
        default_min_num_periods: Optional[int] = None,
    ) -> None:
        """Render a date grouping selector and optional date range filter.

        Displays a selectbox for choosing the date aggregation level and
        optionally renders min/max date selectors to constrain the visible
        range. Updates ``self.date_grouping``, ``self.min_date_grouping``,
        and ``self.max_date_grouping``.

        Args:
            default: Default date grouping level. Falls back to
                ``self.default_date_grouping``.
            default_min: Default minimum date grouping string for the range
                filter.
            default_max: Default maximum date grouping string for the range
                filter.
            default_max_complete_period: If set, automatically compute
                ``default_max`` as the most recent complete period before this
                date.
            default_min_num_periods: If set, automatically compute
                ``default_min`` as this many periods before ``default_max``.
        """
        default = default or self.default_date_grouping
        if default in DATE_GROUPING_MAP:
            index = list(DATE_GROUPING_MAP.keys()).index(default) + 1
        else:
            index = 0
        self.date_grouping = st.selectbox(
            label="Date grouping",
            options=[None, *DATE_GROUPING_MAP.keys()],
            index=index,
        )
        self.add_date_grouping_column()
        if default_max is None and default_max_complete_period is not None:
            default_max = self.get_last_complete_period(default_max_complete_period)
        if default_min is None and default_min_num_periods is not None:
            ref = default_max
            if ref is None:
                # Use latest available value from data
                if (
                    isinstance(self.data, pl.DataFrame)
                    and DATE_GROUPING_COL in self.data.columns
                ):
                    ref = sorted(self.data[DATE_GROUPING_COL].unique().to_list())[-1]
            if ref is not None:
                default_min = self.get_period_offset(ref, default_min_num_periods)
        self.get_date_range_filter(default_min=default_min, default_max=default_max)

    def add_date_grouping_column(self) -> None:
        """Add a ``DateGrouping`` string column to ``self.data``.

        Based on the current ``self.date_grouping`` setting, formats the date
        column into a human-readable period label (e.g., "2024-03", "2024-Q1")
        and stores it as a new column in the data.
        """
        if self.data is None or self.date_col is None:
            return

        self.data = self.data.with_columns(
            _polars_date_grouping_column(self.date_col, self.date_grouping).alias(
                DATE_GROUPING_COL
            )
        )

    def get_last_complete_period(self, dt: date) -> Optional[str]:
        """Return the DateGrouping string for the most recent complete period
        that ended before `dt`, based on self.date_grouping.

        Args:
            dt: The reference date (e.g., date.today()).

        Returns:
            The DateGrouping string, or None if date_grouping is not set.
        """
        if self.date_grouping is None:
            return None

        if self.date_grouping == "Daily":
            d = dt - timedelta(days=1)
            return d.strftime("%Y-%m-%d")

        elif self.date_grouping == "Weekly":
            # Go to the last day of the previous complete week (Sunday before this week's Monday)
            days_since_monday = dt.weekday()  # Monday=0
            last_monday = dt - timedelta(days=days_since_monday)
            # Last complete week ended the Sunday before last_monday
            last_day_prev_week = last_monday - timedelta(days=1)
            return last_day_prev_week.strftime("%G-W%V")

        elif self.date_grouping == "Bi-Weekly":
            # Current ISO week
            w = dt.isocalendar()[1]
            year = dt.isocalendar()[0]
            # Current bi-weekly bucket
            b = (w - 1) // 2 * 2 + 1
            # If we're still in the bucket's 2-week span (week b or b+1),
            # the current bucket isn't complete yet, use previous bucket
            if w <= b + 1:
                b = b - 2
                if b < 1:
                    year = year - 1
                    # Get last ISO week of previous year
                    last_day_prev_year = date(year, 12, 28)
                    last_week = last_day_prev_year.isocalendar()[1]
                    b = (last_week - 1) // 2 * 2 + 1
            return f"{year}-W{b:02d}"

        elif self.date_grouping == "Monthly":
            # First of current month minus 1 day = last day of previous month
            first_of_month = dt.replace(day=1)
            last_day_prev_month = first_of_month - timedelta(days=1)
            return last_day_prev_month.strftime("%Y-%m")

        elif self.date_grouping == "Quarterly":
            # Current quarter
            current_q = (dt.month - 1) // 3 + 1
            # Previous complete quarter
            if current_q == 1:
                return f"{dt.year - 1}-Q4"
            else:
                return f"{dt.year}-Q{current_q - 1}"

        elif self.date_grouping == "Yearly":
            return str(dt.year - 1)

        return None

    def get_period_offset(self, period_str: str, num_periods: int) -> Optional[str]:
        """Return the DateGrouping string that is `num_periods` before `period_str`.

        Args:
            period_str: A DateGrouping string (e.g., "2026-W15", "2026-04").
            num_periods: Number of periods to step back.

        Returns:
            The DateGrouping string offset by num_periods, or None if date_grouping
            is not set.
        """
        if self.date_grouping is None:
            return None

        if self.date_grouping == "Daily":
            dt = date.fromisoformat(period_str)
            result = dt - timedelta(days=num_periods)
            return result.strftime("%Y-%m-%d")

        elif self.date_grouping == "Weekly":
            # Parse "YYYY-WVV"
            year, week = int(period_str[:4]), int(period_str.split("W")[1])
            # Get the Monday of that week, then subtract num_periods weeks
            jan4 = date(year, 1, 4)  # Jan 4 is always in ISO week 1
            monday_w1 = jan4 - timedelta(days=jan4.weekday())
            target_monday = (
                monday_w1 + timedelta(weeks=week - 1) - timedelta(weeks=num_periods)
            )
            return target_monday.strftime("%Y-W%V")

        elif self.date_grouping == "Bi-Weekly":
            # Parse "YYYY-WVV" (bi-weekly bucket start)
            year, week = int(period_str[:4]), int(period_str.split("W")[1])
            jan4 = date(year, 1, 4)
            monday_w1 = jan4 - timedelta(days=jan4.weekday())
            target_monday = (
                monday_w1 + timedelta(weeks=week - 1) - timedelta(weeks=num_periods * 2)
            )
            # Recompute bi-weekly bucket for the target date
            iso_year, iso_week, _ = target_monday.isocalendar()
            b = (iso_week - 1) // 2 * 2 + 1
            return f"{iso_year}-W{b:02d}"

        elif self.date_grouping == "Monthly":
            # Parse "YYYY-MM"
            year, month = int(period_str[:4]), int(period_str[5:7])
            result = date(year, month, 1) - relativedelta(months=num_periods)
            return result.strftime("%Y-%m")

        elif self.date_grouping == "Quarterly":
            # Parse "YYYY-QN"
            year, quarter = int(period_str[:4]), int(period_str[-1])
            total_quarters = year * 4 + quarter - num_periods
            result_year = (total_quarters - 1) // 4
            result_quarter = total_quarters - result_year * 4
            return f"{result_year}-Q{result_quarter}"

        elif self.date_grouping == "Yearly":
            year = int(period_str)
            return str(year - num_periods)

        return None

    def get_date_range_filter(
        self,
        default_min: Optional[str] = None,
        default_max: Optional[str] = None,
    ) -> None:
        """Render Min/Max Date selectboxes based on available DateGrouping values.
        The selected range is used to automatically filter data in update_figure().

        Args:
            default_min: Default value for the Min Date selectbox. If None, defaults to
                the earliest available date grouping.
            default_max: Default value for the Max Date selectbox. If None, defaults to
                the latest available date grouping.
        """
        if self.data is None or self.date_grouping is None:
            return
        if DATE_GROUPING_COL not in self.data.columns:
            self.add_date_grouping_column()
        options = sorted(self.data[DATE_GROUPING_COL].unique().to_list())
        if not options:
            return
        min_index = (
            options.index(default_min) if default_min and default_min in options else 0
        )
        cols = st.columns(2)
        self.min_date_grouping = cols[0].selectbox(
            label="Min Date",
            options=options,
            index=min_index,
            key=f"{self.id}_min_date_grouping",
        )
        options_desc = list(reversed(options))
        max_index = (
            options_desc.index(default_max)
            if default_max and default_max in options_desc
            else 0
        )
        self.max_date_grouping = cols[1].selectbox(
            label="Max Date",
            options=options_desc,
            index=max_index,
            key=f"{self.id}_max_date_grouping",
        )

    def get_options(self) -> None:
        """Render all chart option widgets in the Streamlit UI.

        Creates the header row, axis selectors (x, y, color), and an options
        popover containing graph type, facets, size, bar mode, marginals,
        height, orientation, and legend sorting controls. Sets instance
        attributes (``self.graph_type``, ``self.y``, ``self.x``, ``self.color``,
        etc.) based on user selections.
        """
        cols = self.header(self.title)
        cols[4].write(self._popover_chart_options_style(), unsafe_allow_html=True)
        options_popup = cols[4].popover("Options")

        self.graph_type = options_popup.selectbox(
            label="Graph Type",
            options=self.graph_types,
            index=self.graph_types.index(self.default_graph_type),
            key=f"{self.id}_graph_type",
        )

        self.y = cols[1].selectbox(
            label="y",
            options=self.y_opts,
            index=self.y_opts.index(self.default_y),
            key=f"{self.id}_y",
        )
        self.x = cols[2].selectbox(
            label="x",
            options=self.x_opts,
            index=self.x_opts.index(self.default_x),
            key=f"{self.id}_x",
        )
        self.color = cols[3].selectbox(
            label="Color",
            options=self.color_opts,
            index=self.color_opts.index(self.default_color),
            key=f"{self.id}_color",
            disabled=self.graph_type == "donut",
        )
        _is_grouped_stacked = (
            st.session_state.get(f"{self.id}_barmode") == "grouped+stacked"
        )
        facet_opts = [None] + self.color_opts
        self.facet_col_index = (
            0
            if self.default_facet_col is None
            else facet_opts.index(self.default_facet_col)
        )
        self.facet_col = options_popup.selectbox(
            label="Column Split",
            options=facet_opts,
            index=self.facet_col_index,
            key=f"{self.id}_facet_col",
            disabled=_is_grouped_stacked,
        )
        facet_row_opts = [None] + self.color_opts
        self.facet_row_index = (
            0
            if self.default_facet_row is None
            else facet_row_opts.index(self.default_facet_row)
        )
        self.facet_row = options_popup.selectbox(
            label="Row Split",
            options=facet_row_opts,
            index=self.facet_row_index,
            key=f"{self.id}_facet_row",
            disabled=_is_grouped_stacked,
        )
        size_opts = [None] + self.size_opts
        self.size_index = (
            0 if self.default_size is None else size_opts.index(self.default_size)
        )
        self.size = options_popup.selectbox(
            label="Size",
            options=size_opts,
            index=self.size_index,
            key=f"{self.id}_size",
            disabled=self.graph_type not in ["scatter", "map"],
        )

        self.barmode = options_popup.selectbox(
            label="Bar Mode",
            options=list(BARMODES.keys()),
            index=list(BARMODES.keys()).index(self.default_barmode),
            key=f"{self.id}_barmode",
            disabled=self.graph_type != "bar",
        )
        bar_group_opts = [None] + self.color_opts
        self.bar_group_index = (
            0
            if self.default_bar_group is None
            else bar_group_opts.index(self.default_bar_group)
        )
        self.bar_group = options_popup.selectbox(
            label="Bar Group",
            options=bar_group_opts,
            index=self.bar_group_index,
            key=f"{self.id}_bar_group",
            disabled=self.barmode != "grouped+stacked" or self.graph_type != "bar",
        )
        marginal_opts = [None, "box", "histogram", "rug", "violin"]
        self.marginal_index = (
            0
            if self.default_marginal is None
            else marginal_opts.index(self.default_marginal)
        )
        self.marginal = options_popup.selectbox(
            label="Marginal Plots",
            options=marginal_opts,
            index=self.marginal_index,
            key=f"{self.id}_marginal",
            disabled=self.graph_type != "scatter",
        )
        self.histogram_bins = options_popup.number_input(
            label="Marginal Histogram Bins",
            value=self.default_histogram_bins,
            step=5,
            key=f"{self.id}_histogram_bins",
        )

        self.height = options_popup.number_input(
            label="Height",
            value=self.default_height,
            min_value=100,
            max_value=2000,
            step=25,
            key=f"{self.id}_height",
        )

        self.orientation_h = options_popup.checkbox(
            label="Horizontal bars",
            value=self.default_orientation_h,
            key=f"{self.id}_orientation",
            disabled=self.graph_type != "bar" or self.x in HORIZONTAL_BAR_BLOCKED_COLS,
        )
        self.orientation = "h" if self.orientation_h else "v"
        self.sort_legend_by_value = options_popup.checkbox(
            label="Sort legend by value",
            value=self.default_sort_legend_by_value,
            disabled=True if self.facet_col or self.facet_row else False,
            key=f"{self.id}_sort_legend_by_value",
        )

    @staticmethod
    def _filter_date_range(
        df: pl.DataFrame,
        min_period: str,
        max_period: str,
    ) -> pl.DataFrame:
        """Keep rows whose DateGrouping label falls within the inclusive range.
        Label formats are zero-padded, so lexicographic order matches chronology.
        """
        return df.filter(
            (pl.col(DATE_GROUPING_COL) >= min_period)
            & (pl.col(DATE_GROUPING_COL) <= max_period)
        )

    def update_figure(
        self,
        orientation: Optional[str] = None,
        colormaps: Optional[Dict[str, Any]] = None,
        map_theme: Optional[str] = None,  # Light or Dark
        **kwargs: Any,
    ) -> None:
        """Build the Plotly figure based on current widget selections.

        Performs date grouping/aggregation, applies date range filters, then
        dispatches to the appropriate graph factory (bar/line/scatter, donut,
        sunburst, or map). The resulting figure is stored in ``self.fig``.

        Args:
            orientation: Override orientation ("h" or "v"). Defaults to the
                widget-selected orientation.
            colormaps: Override color maps. Defaults to ``self.colormaps``.
            map_theme: Map tile theme — "Light" or "Dark". Only applies when
                ``graph_type`` is "map".
            **kwargs: Additional keyword arguments passed to the underlying
                graph factory function.
        """
        self.data_chart = self.group_by_date(
            self.data,
            date_grouping=self.date_grouping,
            date_col=self.date_col,
            grp_col=self.color,
            extra_grp_cols=(
                [self.bar_group]
                if self.bar_group and self.barmode == "grouped+stacked"
                else None
            ),
        )
        if (
            self.date_grouping
            and self.min_date_grouping is not None
            and self.max_date_grouping is not None
            and DATE_GROUPING_COL in self.data_chart.columns
        ):
            self.data_chart = self._filter_date_range(
                self.data_chart, self.min_date_grouping, self.max_date_grouping
            )
        df = self.data_chart
        if self.graph_type == "map":
            if "lat" not in df.columns or "lon" not in df.columns:
                st.error("Map requires lat and lon columns")
                return

            self.data_nomap = df.filter((col("lat").is_null()) | (col("lon").is_null()))
            df = df.filter(~(col("lat").is_null()) & ~(col("lon").is_null()))
            if self.size:
                self.data_nosize = df.filter(
                    (col(self.size).is_null()) | (col(self.size) <= 0)
                )
                df = df.filter(~(col(self.size).is_null()) & ~(col(self.size) <= 0))

            self.fig = graphs.map(
                df,
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
                df,
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
            color_discrete_map = colormaps.get(self.color) if colormaps else None
            self.fig = graphs.sunburst(
                df,
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
                df,
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
                bar_group=self.bar_group if self.barmode == "grouped+stacked" else None,
                **marginal_args,
                **kwargs,
            )

            if self.graph_type == "scatter" and self.marginal == "histogram":
                self.fig = self.fig.update_traces(
                    nbinsx=self.histogram_bins,
                    nbinsy=self.histogram_bins,
                    selector=dict(type="histogram"),
                )

    def show_figure(self, width: str = "stretch") -> None:
        """Display the Plotly figure in the Streamlit app.

        Args:
            width: The width of the chart. ``"stretch"`` (default) fills the
                container width. ``"content"`` uses the figure's native width.
                An integer pixel value is also accepted.

        Raises:
            Displays a Streamlit error if :meth:`update_figure` has not been called.
        """
        if self.fig is not None:
            st.plotly_chart(
                self.fig,
                width=width,
                key=f"{self.id}_plotly_chart",
            )
        else:
            st.error("Figure is not updated. Call Chart.update_figure() first.")

    @staticmethod
    def data_expander(df: pl.DataFrame, title: str, **kwargs: Any) -> None:
        """Render a DataFrame inside a Streamlit expander with row count.

        Args:
            df: The Polars DataFrame to display.
            title: Label for the expander header.
            **kwargs: Additional arguments passed to ``st.expander()``.
        """
        with st.expander(f"{title} ({df.shape[0]} records)", **kwargs):
            st.dataframe(df)

    def show_data(
        self,
        raw_data: bool = True,
        chart_data: bool = False,
        map_data: bool = True,
        sort_col: Optional[str] = None,
        sort_desc: bool = False,
        **kwargs: Any,
    ) -> None:
        """Display data tables in expandable sections below the chart.

        Args:
            raw_data: If True, show the full source data in an expander.
            chart_data: If True, show the grouped/aggregated chart data.
            map_data: If True and graph_type is "map", show rows excluded from
                the map due to missing lat/lon or size values.
            sort_col: Column name to sort all displayed tables by.
            sort_desc: If True, sort descending. Defaults to False (ascending).
            **kwargs: Additional arguments passed to ``st.expander()``.
        """
        data = self.data
        data_chart = self.data_chart
        if sort_col is not None:
            if isinstance(data, pl.DataFrame) and sort_col in data.columns:
                data = data.sort(sort_col, descending=sort_desc)
            if isinstance(data_chart, pl.DataFrame) and sort_col in data_chart.columns:
                data_chart = data_chart.sort(sort_col, descending=sort_desc)
        if raw_data:
            self.data_expander(data, f"{self.title} data", **kwargs)
        if chart_data:
            self.data_expander(data_chart, f"{self.title} chart data", **kwargs)
        if map_data and self.graph_type == "map":
            self.data_expander(
                self.data_nomap, f"{self.title} data missing lat/lon", **kwargs
            )
            self.data_expander(
                self.data_nosize, f"{self.title} data missing {self.size}", **kwargs
            )

    def highlight_monthly_regions(
        self,
        grouping: str,
        min_date: date,
        max_date: date,
        min_month_chart: date,
        font_color: str = "white",
        fillcolor: str = "#888888",
    ) -> None:
        """
        Alternate background color for each period in the chart.

        Args:
            grouping: Year, Fiscal Year, Half-Year, Fiscal Half-Year, Quarter,
                Fiscal Quarter

        Returns:
            Updates self.fig with vrects and annotations for each period.
        """
        if self.graph_type != "bar":
            return

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
            return

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
                    text="",
                )
            )

        if add_final_period:
            periods.append(
                dict(
                    x0=dates[-1],
                    x1=max_date - relativedelta(day=15),
                    fillcolor=fillcolor if len(periods) % 2 == 0 else None,
                    text="<br>*INCOMPLETE PERIOD*",
                )
            )

        # loop periods and add vrects/text
        # --------------------------------
        max_val = (
            self.data.group_by(MONTH_COL)
            .agg(pl.sum(self.y))
            .select(self.y)
            .max()
            .item()
        )
        prev_val = None
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

            period["title"] = (
                f'{period["x0"] + relativedelta(months=1, day=1)} – {period["x1"] + relativedelta(day=31)}'
            )

            val = (
                self.data.filter(
                    (col(MONTH_COL) >= period["x0"] + relativedelta(months=1, day=1))
                    & (col(MONTH_COL) <= period["x1"] + relativedelta(day=31))
                )
                .select(pl.sum(self.y))
                .item()
            )

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
                valdiff = val - prev_val
                valdiffpct = valdiff / prev_val if prev_val != 0 else 1
                text = f"{valdiff:+,} | {valdiffpct:+.1%}"
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

            prev_val = val
