"""Interactive filter widgets for Streamlit apps with Polars DataFrame support.

This module provides the :class:`Filter` class for building reusable,
cacheable filter UIs in Streamlit. Filters can be applied to Polars
DataFrames via :func:`filter_data` or converted to SQL WHERE clauses.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import polars as pl
import streamlit as st
from polars import col

from chartly import utils


class Filter:
    """A collection of interactive Streamlit filter widgets.

    Create a ``Filter``, add selectbox or multiselect widgets, then apply
    the filter to a DataFrame using :func:`filter_data` or generate SQL
    with :meth:`filter_sql`.

    Example:
        >>> from chartly import Filter, filter_data
        >>> flt = Filter(id="my_filters")
        >>> flt.multiselect("Region", options=["North", "South", "East", "West"])
        >>> filtered_df = filter_data(df, flt)
    """
    def __init__(
        self,
        id: str,
        filters: Optional[Dict[str, Any]] = None,
        label_visibility: str = "visible",
    ) -> None:
        """
        Args:
            id: A unique identifier for the filter. This is required for using the
                session state to cache filter values from multiple instances of Filter.
                It should be unique to all the Filter instances in the app.
            filters: A dictionary of filters.
        """
        self.id = id
        self.filters = filters or {}
        self.label_visibility = label_visibility  # visible, collapsed, or hidden
        self.col_names = {}
        self.filter_types = {}
        self.bypass_options = {}

    def selectbox(
        self,
        label: str,
        options: List[Any],
        default: Any = None,
        col_name: Optional[str] = None,
        placeholder: Optional[str] = None,
        label_visibility: Optional[str] = None,
        filter_type: str = "eq",  # eq, gte, lte, gt, lt
        bypass_option: Any = None,
        **kwargs: Any,
    ) -> None:
        """Add a single-select filter using a Streamlit selectbox widget.

        Args:
            label: Display label for the widget (also used as the filter key).
            options: List of selectable values.
            default: Default selected value. Defaults to the first option.
            col_name: DataFrame column name to filter on. Defaults to ``label``.
            placeholder: Placeholder text when no value is selected.
            label_visibility: Streamlit label visibility ("visible", "collapsed",
                "hidden").
            filter_type: Comparison operator — "eq", "gte", "lte", "gt", or "lt".
            bypass_option: If the selected value equals this, the filter is
                skipped (acts as an "All" option).
            **kwargs: Additional arguments passed to ``st.selectbox()``.
        """
        self.col_names[label] = col_name or label
        default = default or options[0]
        key = f"{self.id}_{label}_{filter_type}"
        self.filter_types[label] = filter_type
        self.bypass_options[label] = bypass_option
        self.filters[label] = st.selectbox(
            label=label,
            options=options,
            index=options.index(st.session_state.get(key) or default),
            placeholder=placeholder or label,
            label_visibility=label_visibility or self.label_visibility,
            key=key,
            **kwargs,
        )

    def multiselect(
        self,
        label: str,
        options: List[Any],
        default: Optional[List[Any]] = None,
        col_name: Optional[str] = None,
        placeholder: Optional[str] = None,
        label_visibility: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a multi-select filter using a Streamlit multiselect widget.

        When values are selected, :func:`filter_data` will keep only rows
        where the column value is in the selected list.

        Args:
            label: Display label for the widget (also used as the filter key).
            options: List of selectable values.
            default: Default selected values. Defaults to empty (no filter).
            col_name: DataFrame column name to filter on. Defaults to ``label``.
            placeholder: Placeholder text when no values are selected.
            label_visibility: Streamlit label visibility ("visible", "collapsed",
                "hidden").
            **kwargs: Additional arguments passed to ``st.multiselect()``.
        """
        default = default or []
        self.col_names[label] = col_name or label
        key = f"{self.id}_{label}"
        self.filters[label] = st.multiselect(
            label=label,
            options=options,
            default=st.session_state.get(key) or default,
            placeholder=placeholder or label,
            label_visibility=label_visibility or self.label_visibility,
            key=key,
            **kwargs,
        )

    @staticmethod
    def list(items: List[str]) -> str:
        """Takes a python list and returns a SQL list.

        Args:
            items: A python list.

        Examples:
            >>> items = ["a", "b", "c"]

            >>> self.list(items)
            ('a','b','c')

            >>> self.read(f"select * from table where col in {self.list(items)}")
            select * from table where col in ('a','b','c')

        """
        return "('" + "','".join(items) + "')"

    def filter_sql(self, where_or_and: str = "WHERE", prefix: str = "") -> str:
        """Generate a SQL WHERE clause from the active filter selections.

        Only filters with non-empty selections are included. Each filter
        becomes a ``column IN (...)`` condition joined by ``AND``.

        Args:
            where_or_and: SQL keyword to prepend ("WHERE" or "AND").
            prefix: Optional table alias prefix (e.g., "t" becomes "t.column").

        Returns:
            A SQL string like ``WHERE col1 IN ('a','b') AND col2 IN ('x')``,
            or an empty string if no filters are active.
        """
        # Make sure prefix has a trailing dot
        if prefix is not None and prefix != "":
            prefix = prefix + "." if prefix[-1] != "." else prefix
        # Only include filter if it is not empty
        filter_list = [
            (
                f"{prefix}{col_name} in {self.list([str(x) for x in values])}"
                if values != []
                else None
            )
            for col_name, values in self.filters.items()
        ]
        filter_list = list(filter(lambda x: x is not None, filter_list))
        if filter_list == []:
            return ""
        return where_or_and + " " + " and ".join(filter_list)


def hash_func(obj: Filter) -> str:
    """Hash function for Streamlit caching of Filter objects.

    Args:
        obj: A Filter instance.

    Returns:
        The SQL representation of the filter state, used as a cache key.
    """
    return obj.filter_sql()


def combine_filters(filter1: Optional[Filter], filter2: Optional[Filter]) -> Optional[Filter]:
    """Combine two filters into a single filter.
    If same key exists in both, prefer the first filter, unless it is empty then use the
    second filter.
    """
    if filter1 is None:
        return filter2
    elif filter2 is None:
        return filter1
    f1 = filter1.filters
    f2 = filter2.filters
    filters = dict()
    for col_name in list(set(list(f1.keys()) + list(f2.keys()))):
        if (
            col_name in f1.keys()
            and f1.get(col_name) is not None
            and f1.get(col_name) != []
        ):
            filters[col_name] = f1.get(col_name)
        elif (
            col_name in f2.keys()
            and f2.get(col_name) is not None
            and f2.get(col_name) != []
        ):
            filters[col_name] = f2.get(col_name)
    flt = Filter(
        id=filter1.id + "_" + filter2.id,
        filters=filters,
    )
    return flt


@st.cache_data(ttl=None, hash_funcs={Filter: hash_func, pl.DataFrame: utils.pl2pd})
def filter_data(
    df: pl.DataFrame,
    flt: Filter,
    names: Optional[Union[str, List[str]]] = None,
    return_size_too: bool = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, int]]]:
    """Apply a Filter's selections to a Polars DataFrame (cached).

    Iterates through the filter's active selections and progressively
    filters the DataFrame. Results are cached by Streamlit to avoid
    redundant computation on reruns.

    Args:
        df: The source Polars DataFrame to filter.
        flt: A :class:`Filter` instance with active selections.
        names: Subset of filter keys to apply. If None, all filters are applied.
            Can be a single string or list of strings.
        return_size_too: If True, also return a dictionary mapping each filter
            step name to the row count after that step.

    Returns:
        The filtered DataFrame, or a tuple of (filtered DataFrame, size dict)
        if ``return_size_too`` is True.
    """
    df_size = {"Total": df.shape[0]}
    if names is None:
        names = flt.filters.keys()
    elif not isinstance(names, list):
        names = [names]
    for k, v in flt.filters.items():
        col_name = flt.col_names.get(k)
        bypass_option = flt.bypass_options.get(k)
        filter_type = flt.filter_types.get(k)
        if col_name is not None and col_name in df.columns and k in names:
            if isinstance(v, list):
                if v != []:
                    df = df.filter(col(col_name).is_in(v))
            elif v != bypass_option and filter_type is not None:
                if filter_type == "eq":
                    df = df.filter(col(col_name) == v)
                elif filter_type == "gte":
                    df = df.filter(col(col_name) >= v)
                elif filter_type == "lte":
                    df = df.filter(col(col_name) <= v)
                elif filter_type == "gt":
                    df = df.filter(col(col_name) > v)
                elif filter_type == "lt":
                    df = df.filter(col(col_name) < v)
            df_size[k] = df.shape[0]
    if return_size_too:
        return df, df_size
    return df
