
import polars as pl

from utils.types import (
    FreqType,
)
from utils.utils import flatten

# features
from .group_trades import *

prod_factors = {
    'tpc': expr_tpc,
    'btpup': expr_btpup,
    'stpdown': expr_stpdown,
    'tpcskew': expr_tpcskew,
    'rsj': expr_rsj
}

def generate_bars_from_strades(
    data: pl.DataFrame,
    freq: FreqType,
    dtname: str,
    ) -> pl.DataFrame:
    return data.sort(dtname).group_by_dynamic( # old version: groupby_dynamic
            index_column=dtname,
            every=freq,
            label='right',
        ).agg(
            flatten(
                (
                    (
                        pl.col('price').first().alias('open'),
                        pl.col('price').max().alias('high'),
                        pl.col('price').last().alias('close'),
                        pl.col('price').min().alias('low'),
                        pl.col('amount').sum().alias('volume'),
                        (pl.col('amount') * pl.col('price')).sum().alias('quote_volume'),
                        pl.col('amount').filter(pl.col('side') == 'buy').sum().alias('taker_buy_volume'),
                        (pl.col('amount') * pl.col('price')).filter(pl.col('side') == 'buy').sum().alias('taker_buy_quote_volume'),
                        pl.len().alias('count')
                    ),
                    (
                        expr().alias(fname) for fname, expr in prod_factors.items()
                    )
                )
            )
        )

