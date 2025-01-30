
# oi stats: var, kurtosis, skew, upvar, ratio_upvar, trendratio, max drawdown (shadow)
# oi, ret/amt corr


import polars as pl
import numpy as np

def expr_moito(oiname, amtname, window = 72):
    return (pl.col(oiname) / pl.col(amtname).rolling_mean(window)).rolling_mean(window)

def expr_oiskew(oiname, window = 72):
    return pl.col(oiname).pct_change().rolling_skew(window)

def expr_oistd(oiname, window = 72):
    return pl.col(oiname).pct_change().rolling_std(window)

def expr_oitrendratio(oiname, window = 72):
    doi = pl.col(oiname).pct_change()
    return doi.rolling_sum(window) / doi.abs().rolling_sum(window)
    
def expr_oidrop(oiname, window = 168):
    ath = pl.col(oiname).rolling_max(window)
    return (ath - pl.col(oiname)) / ath

def expr_oiretcorr(oiname, cname, window = 8):
    doi = pl.col(oiname).pct_change()
    ret = pl.col(cname).pct_change()
    return pl.rolling_corr(ret,doi, window_size = window)

def expr_oiautocorr(oiname, window = 672):
    doi = pl.col(oiname).pct_change()
    bf = doi.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_oilevelautocorr(oiname, window = 672):
    bf = pl.col(oiname).rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

#%%
def expr_oiwp(oiname, cname, window=20):
    return (pl.col(cname) * pl.col(oiname)).rolling_mean(window) /  pl.col(oiname).rolling_mean(window)

def expr_oiamtcorr(oiname, aname, window=5):
    return pl.rolling_corr(pl.col(aname).diff(),pl.col(oiname).diff(), window_size = window)

def expr_amtdivergence(oiname, cname, window=10):
    vol_ma = pl.col(cname).rolling_mean(window)
    oi_ma = pl.col(oiname).rolling_mean(window)
    return (vol_ma / oi_ma).pct_change(window)

def expr_oito(aname, window=20):
    return (
        pl.col(aname).rolling_sum(window) / 
        (pl.col().rolling_mean(window) + 1e-9)
    )

