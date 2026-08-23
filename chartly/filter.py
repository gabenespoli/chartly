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
        if default is None:
            default = options[0]
        self.col_names[label] = col_name or label
        key = f"{self.id}_{label}_{filter_type}"
        self.filter_types[label] = filter_type
        self.bypass_options[label] = bypass_option
        self.filters[label] = st.selectbox(
            label=label,
            options=options,
            index=options.index(self._stored_selection(key, default)),
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
        if default is None:
            default = []
        self.col_names[label] = col_name or label
        key = f"{self.id}_{label}"
        self.filters[label] = st.multiselect(
            label=label,
            options=options,
            default=self._stored_selection(key, default),
            placeholder=placeholder or label,
            label_visibility=label_visibility or self.label_visibility,
            key=key,
            **kwargs,
        )

    @staticmethod
    def _stored_selection(key: str, fallback: Any) -> Any:
        """Return the user's stored widget selection for key, honoring falsy values
        like 0 or an empty list instead of falling back to the default."""
        return st.session_state[key] if key in st.session_state else fallback

    @staticmethod
    def list(items: List[str]) -> str:
        """Takes a python list and returns a SQL list.

        Single quotes inside values are escaped by doubling so they cannot break
        out of the quoted literal.

        Args:
            items: A python list.

        Examples:
            >>> items = ["a", "b", "c"]

            >>> self.list(items)
            ('a','b','c')

            >>> self.read(f"select * from table where col in {self.list(items)}")
            select * from table where col in ('a','b','c')

        """
        escaped = [item.replace("'", "''") for item in items]
        return "('" + "','".join(escaped) + "')"

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
        filter_list = []
        for label, value in self.filters.items():
            if value == []:
                continue
            if not isinstance(value, list):
                # selectbox filters store a scalar instead of a list
                value = [value]
            filter_list.append(
                f"{prefix}{label} in {self.list([str(x) for x in value])}"
            )
        if filter_list == []:
            return ""
        return where_or_and + " " + " and ".join(filter_list)


def filter_hash(obj: Filter) -> str:
    """Hash function for Streamlit caching of Filter objects.

    Args:
        obj: A Filter instance.

    Returns:
        The SQL representation of the filter state, used as a cache key.
    """
    return obj.filter_sql()
    return obj.filter_sql()


def combine_filters(
    filter1: Optional[Filter], filter2: Optional[Filter]
) -> Optional[Filter]:
    """Combine two filters into a single filter.
    If same key exists in both, prefer the first filter, unless it is empty then use the
    second filter. Metadata (col_names, filter_types, bypass_options) follows whichever
    filter supplied each label, so the combined filter works with filter_data.
    """
    if filter1 is None:
        return filter2
    elif filter2 is None:
        return filter1
    f1 = filter1.filters
    f2 = filter2.filters
    filters = dict()
    col_names = dict()
    filter_types = dict()
    bypass_options = dict()
    for label in list(set(list(f1.keys()) + list(f2.keys()))):
        if label in f1.keys() and f1.get(label) is not None and f1.get(label) != []:
            source = filter1
        elif label in f2.keys() and f2.get(label) is not None and f2.get(label) != []:
            source = filter2
        else:
            continue
        filters[label] = source.filters[label]
        col_names[label] = source.col_names.get(label, label)
        filter_types[label] = source.filter_types.get(label)
        bypass_options[label] = source.bypass_options.get(label)
    flt = Filter(
        id=filter1.id + "_" + filter2.id,
        filters=filters,
    )
    flt.col_names = col_names
    flt.filter_types = filter_types
    flt.bypass_options = bypass_options
    return flt


@st.cache_data(ttl=None, hash_funcs={Filter: filter_hash, pl.DataFrame: utils.pl2pd})
def filter_data(
    df: Union[pd.DataFrame, pl.DataFrame],
    flt: Filter,
    names: Optional[Union[str, List[str]]] = None,
    return_size_too: bool = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, int]]]:
    """Apply a Filter's selections to a DataFrame (cached).

    Iterates through the filter's active selections and progressively
    filters the DataFrame. Results are cached by Streamlit to avoid
    redundant computation on reruns.

    Args:
        df: The source DataFrame (Pandas or Polars) to filter. Converted to
            Polars internally.
        flt: A :class:`Filter` instance with active selections.
        names: Subset of filter keys to apply. If None, all filters are applied.
            Can be a single string or list of strings.
        return_size_too: If True, also return a dictionary mapping each filter
            step name to the row count after that step.

    Returns:
        The filtered Polars DataFrame, or a tuple of (filtered DataFrame,
        size dict) if ``return_size_too`` is True.

    Note:
        ``return_size_too`` is kept as a boolean flag deliberately: callers
        rely on st.cache_data hashing this exact call shape; splitting it in
        two would change cache keys. Revisit only on a major version bump.
    """
    df = utils.ensure_polars(df)
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
