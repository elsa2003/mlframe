"""
this is different from group_trades:
    group_trades generate feature for aggregated values e.g. trades->15min,
    group_bar generate features for the same length e.g. 15min -> 15min

"""

import polars as pl
import numpy as np

def expr_macd(cname, window = 9, swindow = 12, lwindow = 26):
    return ((pl.col(cname).ewm_mean(span=swindow) - pl.col(cname).ewm_mean(span=lwindow)) - (pl.col(cname).ewm_mean(span=swindow) - pl.col(cname).ewm_mean(span=lwindow)).ewm_mean(span=window))
    

def expr_smooth(cname, window):
    return pl.col(cname).rolling_mean(window)

def expr_bollingup(cname, window, band = 1):
    return (pl.col(cname).rolling_mean(window) + band * pl.col(cname).rolling_std(window))

# rev 
def expr_rev_vwap(cname, lname, hname, vname, window = 672, swindow = 8, cap = 0.2):
    volp = (pl.col(cname) + pl.col(lname) + pl.col(hname))/3 * pl.col(vname)
    volp2w = volp.rolling_sum(window) / pl.col(vname).rolling_sum(window)
    ma8 = pl.col(cname).rolling_mean(swindow)
    scale = cap - (ma8 / volp2w - 1).abs().clip( upper_bound = cap)
    sign = (ma8 - volp2w).sign()
    return sign * scale

def expr_rev_boll(cname, vname, window = 672, vswindow = 24, vlwindow = 168, awindow = 8, thres = 2.5):
    band = pl.col(cname).rolling_mean(window) 
    uband = band + band.rolling_std(window) * thres
    logv = pl.col(vname).clip(lower_bound = 1).log()
    lvcon = (logv.rolling_mean(vswindow) - logv.rolling_mean(vlwindow)) /logv.rolling_std(vlwindow)
    scale = ((uband - pl.col(cname)).clip( lower_bound = 0) / (uband - band)).abs()
    sign = (pl.col(cname) - band).sign()
    cond = (lvcon > 1).cast(int)
    return scale * sign * cond.rolling_max(awindow)


def expr_rev_candel(oname, cname, hname, lname, window = 4):
    phigh = pl.col(hname).rolling_max(window)
    popen = pl.col(oname).shift(window-1)
    plow = pl.col(lname).rolling_min(window)
    return pl.when(
            (phigh > popen) &
            (pl.col(cname).shift(window) < popen.shift(window)) &
            (phigh - pl.col(cname) > pl.col(cname) - plow)
        ).then(-1)
        .when(
            (pl.col(cname) < popen) &
            (pl.col(cname).shift(window) > popen.shift(window)) &
            (pl.col(cname) - plow > phigh - pl.col(cname))
        ).then(1)
        .otherwise(0)

def expr_rev_amtsupp(cname, hname, lname, amtname, window=168):
    return (
        (pl.col(amtname) * (pl.col(cname) - pl.col(lname))).rolling_sum(window) /
        (pl.col(amtname) * (pl.col(hname) - pl.col(lname))).rolling_sum(window)
    )

def expr_rev_pullback(cname, lname, cwindow=20, rwindow=12):
    plow = pl.col(lname).rolling_min(cwindow)
    return (
        (pl.col(cname) - plow) / (plow + 1e-9)
    ).rolling_mean(rwindow)


def expr_rsi(cname, window = 24):
    pdiff = pl.col(cname).diff()
    upc = pl.when(pdiff < 0).then(pl.lit(0)).otherwise(pdiff)
    dpc = pl.when(pdiff > 0).then(pl.lit(0)).otherwise(pdiff * -1)
    supc = upc.ewm_mean(span = window)
    sdpc = dpc.ewm_mean(span = window)
    return - sdpc / supc


def expr_rev_lbreak(lname, window=24, thres = 0.005):
    support_level = pl.col(lname).rolling_min(window)
    return (
        (pl.col(lname) <= support_level * (1.+thres)).cast(int)
        .rolling_sum(window)
    ) 

def expr_rev_hbreak(hname, window=24, thres = 0.005):
    support_level = pl.col(hname).rolling_max(window)
    return (
        (pl.col(hname) >= support_level * (1.-thres)).cast(int)
        .rolling_sum(window)
    ) 

def expr_zclose(cname, window = 168):
    bolldev = (pl.col(cname) - pl.col(cname).rolling_mean(window))/pl.col(cname).rolling_std(window)
    bolldev = bolldev.clip(lower_bound = -5, upper_bound = 5)
    return bolldev

def expr_lpm(cname, q = 0.75, window = 8):
    ret = pl.col(cname).pct_change()
    lpm = ret.rolling_quantile(quantile = q, interpolation = 'nearest', window_size = window) - ret
    lpm = pl.when(lpm <= 0).then(pl.lit(0)).otherwise(lpm) **2
    return lpm


# ret stats
def expr_amtskew(amtname, window = 8):
    bf = pl.col(amtname).rolling_skew(window) * -1
    return bf

def expr_amtstd(amtname, window = 24):
    bf = pl.col(amtname).rolling_std(window)
    return bf

def expr_trendratio(cname, window = 24):
    ret = pl.col(cname).pct_change()
    return ret.rolling_sum(window) / ret.abs().rolling_sum(window)

def expr_pricemdd(cname, window = 168):
    ath = pl.col(cname).rolling_max(window)
    return (ath - pl.col(cname)) / ath
    
def expr_upvar(cname, window = 168):
    ret = pl.col(cname).pct_change()
    return pl.when(ret > 0).then(ret).otherwise(None).rolling_std(
            window_size=window,
            min_periods=1,         # adjust min_periods if needed
        )

# amt stats
def expr_zamihudilliq(amtname, cname, window = 8, zwindow = 168):
    ret = pl.col(cname).pct_change()
    bf = (ret / pl.col(amtname)).abs().rolling_mean(window) * -1
    return (bf - bf.rolling_mean(zwindow))/bf.rolling_std(zwindow)


def expr_amihudilliqhl(hname, lname, amtname, window=20):
    return (
        (pl.col(hname) - pl.col(lname)).abs() / 
        (pl.col(amtname).rolling_mean(window) + 1e-9)
    )

def expr_mamt(amtname, window = 8):
    return pl.col(amtname).rolling_mean(window)

def expr_amtinflow(cname, amtname, window = 168):
    ret = pl.col(cname).pct_change()
    retup= pl.when(ret > 0).then(ret.pow(2)).otherwise(0).rolling_sum(window)
    retdown= pl.when(ret < 0).then(ret.pow(2)).otherwise(0).rolling_sum(window)
    bf = (retup - retdown) / (retup + retdown) * (pl.col(amtname)/pl.col(amtname).shift(window) - 1)
    return bf

def expr_pvcorr(cname, amtname, window=14):
    price_trend = pl.col(cname).diff(window)
    volume_trend = pl.col(amtname).diff(window)
    return pl.rolling_corr(price_trend, volume_trend, window_size = window)

def expr_amtcum(oname, cname, amtname, window=20):
    return ((pl.col(cname) - pl.col(oname)) * pl.col(amtname)).rolling_sum(window)


def expr_amtcluster(amtname, window=24):
    intra_hour_vol = pl.col(amtname) / pl.col(amtname).rolling_sum(window)
    return -(intra_hour_vol * intra_hour_vol.log()).replace([-float('inf'),float('inf')], 0).rolling_sum(window)

def expr_shadowsz(oname, cname, hname, lname, window=5):
    body_size = (pl.col(cname) - pl.col(oname)).abs()
    shadow_ratio = (pl.col(hname) - pl.col(lname)) / (body_size + 1e-9)
    return shadow_ratio.rolling_mean(window)

def expr_amtresonance(cname, amtname, window=14):
    return (pl.col(cname).pct_change() * pl.col(amtname).pct_change().abs()).rolling_mean(window)


# vol
def expr_volskew(cname, vwindow = 24, window = 72):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    return vol.rolling_skew(window)


def expr_volretcorr(cname, vwindow = 21, window = 8):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    return pl.rolling_corr(ret,vol, window_size = window)

def expr_volautocorr(cname, vwindow = 24, window = 672):
    ret = pl.col(cname).pct_change()
    vol = ret.clip(lower_bound = -0.3, upper_bound=0.3).ewm_std(half_life = vwindow)
    bf = vol.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_volcontraction(hname, lname, window=20):
    range_ratio = (pl.col(hname) - pl.col(lname)) / pl.col(lname)
    return (range_ratio.rolling_std(window) < range_ratio.rolling_mean(window)).cast(int)

def expr_tailr(cname, window=100):
    ret = pl.col(cname).pct_change()
    return (ret.rolling_skew(window) * 0.5 + ret.rolling_kurtosis(window) * 0.5)

def expr_tailc(cname, window=50, z=2):
    ret = pl.col(cname).pct_change()
    return (ret.abs() > z * ret.rolling_std(window)).cast(int).rolling_mean(window)

def expr_retvolr(cname, window=20):
    ret = pl.col(cname).pct_change()
    return ret.rolling_mean(window) / (ret.rolling_std(window) + 1e-9)

def expr_liqrank(window=30):
    dollar_volume = (pl.col('close') * pl.col('volume')).rolling_mean(window)
    return

def expr_retautocorr(cname, window = 672):
    ret = pl.col(cname).pct_change()
    bf = ret.rolling_map(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], window_size = window)
    return bf

def expr_intrabarvolskew(cname, oname, hname, lname, window=20):
    open_to_close = (pl.col(cname)/pl.col(oname)).log()
    high_to_low = (pl.col(hname)/pl.col(lname)).log()
    return (high_to_low.rolling_std(window) - open_to_close.rolling_std(window))
    
def expr_voltailc(cname, window=30, z_threshold=2):
    ret = pl.col(cname).pct_change()
    vol = ret.rolling_std(window)
    return (ret.abs() > z_threshold * vol.shift(1)).cast(int)

def expr_lsvolr(swindow=5, lwindow=60):
    ret = pl.col(cname).pct_change()
    return ret.rolling_std(swindow) / (ret.rolling_std(lwindow) + 1e-9)
    
#-
# bins

# pattern
def expr_pattern_tripletrend(oname, cname):
    cond1 = pl.col(cname) > pl.col(oname)
    cond2 = pl.col(cname).shift(1) > pl.col(oname).shift(1)
    cond3 = pl.col(cname).shift(2) > pl.col(oname).shift(2)
    return (
        (cond1 & cond2 & cond3).cast(int) -
        (~cond1 & ~cond2 & ~cond3).cast(int)
    )

def expr_pattern_wedge(hname, lname, window=20):
    highs = pl.col(hname).rolling_max(window)
    lows = pl.col(lname).rolling_min(window)
    return (
        ((highs - highs.shift(window)) < 
         (lows - lows.shift(window))).cast(int)
    )

def expr_pattern_engulf(oname, cname):
    bull_engulf = (
        (pl.col(cname) > pl.col(oname)) &
        (pl.col(cname) > pl.col(oname).shift(1)) &
        (pl.col(oname) < pl.col(cname).shift(1))
    )
    bear_engulf = (
        (pl.col(cname) < pl.col(oname)) &
        (pl.col(cname) < pl.col(oname).shift(1)) &
        (pl.col(oname) > pl.col(cname).shift(1))
    )
    return (bull_engulf.cast(int) - bear_engulf.cast(int))

def expr_fractal_dim(hname, lname, window=50):
    range_ = pl.col(hname) - pl.col(lname)
    return (range_.rolling_std(window) / range_.rolling_mean(window))
