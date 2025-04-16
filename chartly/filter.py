import pandas as pd
import polars as pl
import streamlit as st
from polars import col

from chartly import utils


class Filter:
    def __init__(self, id: str, filters: dict):
        """
        Args:
            id: A unique identifier for the filter. This is required for using the
                session state to cache filter values from multiple instances of Filter.
                It should be unique to all the Filter instances in the app.
            filters: A dictionary of filters.
        """
        self.id = id
        self.filters = filters

    def multiselect(
        self,
        col_name: str,
        options: list,
        default: list = [],
        label: str = None,
        placeholder: str = None,
        label_visibility: str = "collapsed",
        **kwargs,
    ):
        """Add a filter to the self.filters dictionary using a streamlit multiselect
        widget."""
        label = None if label == "" else label
        key = f"{self.id}_{col_name}"
        self.filters[col_name] = st.multiselect(
            label=label or col_name,
            options=options,
            default=st.session_state.get(key) or default,
            placeholder=placeholder or label or col_name,
            label_visibility=label_visibility,
            key=key,
            **kwargs,
        )

    @staticmethod
    def list(items: list) -> str:
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


def hash_func(obj: Filter):
    return obj.filter_sql()


def combine_filters(filter1: Filter, filter2: Filter) -> Filter:
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
    df: pl.DataFrame, flt: Filter, return_size_too: bool = False
) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)
    df_size = {"Total": df.shape[0]}
    for k, v in flt.filters.items():
        if v is not None and v != []:
            if k in df.columns:
                df = df.filter(col(k).is_in(v))
                df_size[k] = df.shape[0]
    if return_size_too:
        return df, df_size
    return df
