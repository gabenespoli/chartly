import pandas as pd
import polars as pl


def pl2pd(df: pl.DataFrame) -> pd.DataFrame:
    dtypes = {k: v for k, v in zip(df.columns, df.dtypes)}
    date_cols = [k for k, v in dtypes.items() if v == pl.Date]
    df = df.to_pandas()
    for col in date_cols:
        df[col] = df[col].dt.date
    return df
