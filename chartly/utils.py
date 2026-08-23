from typing import Union

import pandas as pd
import polars as pl


def ensure_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """Convert pandas input to polars at the library boundary; pass through
    frames that are already polars."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def pl2pd(df: pl.DataFrame) -> pd.DataFrame:
    dtypes = {k: v for k, v in zip(df.columns, df.dtypes)}
    date_cols = [k for k, v in dtypes.items() if v == pl.Date]
    df = df.to_pandas()
    for col in date_cols:
        df[col] = df[col].dt.date
    return df
