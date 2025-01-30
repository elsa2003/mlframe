
import polars as pl

from utils.types import (
    FreqType,
)
from utils.utils import flatten

# features
from .group_bar import *

prod_factors = {
    'rsj1h': lambda : expr_smooth('rsj', 4),
    'rsj8h': lambda : expr_smooth('rsj', 32),
    'tpc1h': lambda : expr_smooth('tpc', 4),
    'tpc8h': lambda : expr_smooth('tpc', 32),
    'stpdown1h': lambda : expr_smooth('stpdown', 4),
    'stpdown8h': lambda : expr_smooth('stpdown', 32),
    'tpcskew1h': lambda : expr_smooth('tpcskew', 4),
    'tpcskew8h': lambda : expr_smooth('tpcskew', 32),
    # bar
    'macd1h8h': lambda : expr_macd('close', window = 4, swindow = 4, lwindow = 32),
    'macd8h24h': lambda : expr_macd('close', window = 32, swindow = 32, lwindow = 96),
    'revvwap8h600h': lambda : expr_rev_vwap('close', 'low', 'high', 'volume', window = 2400, swindow = 32, cap = 0.05),
    'revvwap8h200h': lambda : expr_rev_vwap('close', 'low', 'high', 'volume', window = 800, swindow = 32, cap = 0.05),
    'revboll24h168h': lambda : expr_rev_boll('close', 'volume', window = 2400, vswindow = 96, vlwindow = 672, awindow = 32, thres = 2.5),
    'rsi4h': lambda : expr_rsi('close', window = 32),
    'rsi24h': lambda : expr_rsi('close', window = 96),
    'zclose': lambda : expr_zclose('close', window = 168),
    'lpm8h': lambda : expr_lpm('close', 0.75, window = 32),
    'amtskew8h': lambda : expr_amtskew('volume', window = 32),
    'amtstd24h': lambda : expr_amtstd('volume', window = 96),
    'trendratio24h': lambda : expr_trendratio('volume', window = 96),
    'pricemdd72h': lambda : expr_pricemdd('close', window = 288),
    'upvar168h': lambda : expr_upvar('close', window = 672),
    'zamihudilliq8h_168h': lambda : expr_zamihudilliq('volume', 'close', window = 32, zwindow = 672),
    'mamt8h' : lambda : expr_mamt('volume', window = 32),
    'amtinflow72h': lambda : expr_amtinflow('close', 'volume', window = 288),
    'volskew24h_168h': lambda : expr_volskew('close', vwindow = 96, window = 288),
    'volretcorr24h_8h': lambda : expr_volretcorr('close', vwindow = 96, window = 32),
    'volautocorr24h_168h': lambda : expr_volautocorr('close', vwindow = 96, window = 2400)
}

def generate_features_from_bar(
    data: pl.DataFrame,
    dtname: str,
    ) -> pl.DataFrame:
    return data.sort(dtname).with_columns(
            flatten(
                (
                        expr().alias(fname) for fname, expr in prod_factors.items()
                )
            )
        )

