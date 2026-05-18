"""Utility functions for DataFrame conversion between Polars and Pandas.

This module provides helper functions used internally by chartly for
converting between Polars and Pandas DataFrames while preserving column
type semantics.
"""

import pandas as pd
import polars as pl


def pl2pd(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame to a Pandas DataFrame, preserving date columns.

    Polars date columns are converted to Python ``datetime.date`` objects in the
    resulting Pandas DataFrame, rather than Pandas ``Timestamp`` objects. This is
    useful for Streamlit's caching mechanism which hashes DataFrames by value.

    Args:
        df: A Polars DataFrame to convert.

    Returns:
        A Pandas DataFrame with date columns stored as ``datetime.date`` objects.

    Example:
        >>> import polars as pl
        >>> from chartly.utils import pl2pd
        >>> df = pl.DataFrame({"date": [pl.date(2024, 1, 1)], "value": [42]})
        >>> pdf = pl2pd(df)
        >>> type(pdf["date"].iloc[0])
        <class 'datetime.date'>
    """
    dtypes = {k: v for k, v in zip(df.columns, df.dtypes)}
    date_cols = [k for k, v in dtypes.items() if v == pl.Date]
    df = df.to_pandas()
    for col in date_cols:
        df[col] = df[col].dt.date
    return df
