import polars as pl

def expr_tpc():
    return (pl.when(pl.col('price')==pl.col('price').shift(1))
        .then(pl.lit(0))
        .when(pl.col('price')>pl.col('price').shift(1))
        .then(pl.lit(1))
        .when(pl.col('price')<pl.col('price').shift(1))
        .then(pl.lit(-1))
        .otherwise(None)).sum()

def expr_btpup():
    return (pl.when(pl.col('price')>pl.col('price').shift(1))
        .then(1)
        .otherwise(0) * pl.col("side").replace({"buy": 1, "sell": 0}).cast(float)).sum()

def expr_stpdown():
    return (pl.when(pl.col('price')<pl.col('price').shift(1))
        .then(1)
        .otherwise(0) * pl.col("side").replace({"buy": 0, "sell": 1}).cast(float)).sum()

def expr_tpcmax(lookback = 1):
    mpr = ((pl.col('price')-pl.col('price').shift(lookback))/pl.col('price').shift(lookback))
    return mpr.max()

def expr_tpcskew(lookback = 1):
    mpr = ((pl.col('price')-pl.col('price').shift(lookback))/pl.col('price').shift(lookback))
    return ((mpr - mpr.mean())/mpr.std()).pow(3).mean()

def expr_rsj(lookback = 1):
    mpr = ((pl.col('price')-pl.col('price').shift(lookback))/pl.col('price').shift(lookback))
    pv = mpr.filter(mpr > 0).std()
    nv = mpr.filter(mpr < 0).std()
    return (pv - nv) / (pv + nv)

#
def expr_tpcv():
    return (pl.when(pl.col('price') > pl.col('price').shift(1))
            .then(pl.col('amount'))
            .when(pl.col('price') < pl.col('price').shift(1))
            .then(-pl.col('amount'))
            .otherwise(0)).sum()

def expr_vwtd(lookback = 1):
    tpc = pl.col('price').diff(lookback).sign()  # Price change direction
    return (tpc * pl.col('amount')).sum()

def expr_amtacc(lookback = 1):
    return pl.col('amount').diff(lookback).diff(lookback).mean()

def expr_largeoratio(threshold=1):
    big_trades = (pl.col('amount') >= threshold).cast(int)
    return big_trades.mean()

def expr_tickpressure(window=10):
    buy_ticks = (pl.col('price') > pl.col('price').shift(1)).cast(int)
    sell_ticks = (pl.col('price') < pl.col('price').shift(1)).cast(int)
    return (buy_ticks - sell_ticks).mean()

def expr_avolrvolratio(window=20):
    vol_ret = pl.col('price').diff().log().replace([float("inf"), float("-inf")], 0).abs()
    return pl.col('amount').std() / (vol_ret.std() + 1e-9)

def expr_psqrtamt():
    return (
        pl.col('price').diff().abs() / 
        (pl.col('amount').pow(0.5) + 1e-9)
    ).mean()

def expr_vwapdisp(window=20):
    vwap = (pl.col('price') * pl.col('amount')).sum() / pl.col('amount').sum()
    return (pl.col('price') - vwap).abs().mean()

def expr_micint(window=10):
    return (pl.col('amount') * pl.col('price').diff().abs()).mean()

def expr_liqdemand():
    price_change = pl.col('price').diff()
    return (pl.col('amount') * price_change.abs()).sum() / pl.col('amount').sum()

def expr_signautocorr(lag=5):
    sign = pl.col('price').diff().sign()
    return pl.corr(sign, sign.shift(lag))
