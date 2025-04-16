"""
"""
import sys
from datetime import timedelta, datetime, time
import numpy as np
import os
import glob
import pandas as pd
import polars as pl
from time import sleep
import cfg
#from spbook15m.support import *
#from spbook.subopt import run_strat_weights,run_comb_mn
import pytz
import gc
#from cropt2 import run_opt
#from santsignals.sant_signals import *
UTC = pytz.timezone('UTC')

def az(sgl, freq = '1y', nbar1d = 25):
    pnl = (sgl.shift() * ret).sum(axis=1)
    to = sgl.diff().abs().sum(axis=1)
    lev = sgl.abs().sum(axis=1)
    dd = (pnl.cumsum().expanding().max() - pnl.cumsum())
    df = pd.DataFrame(columns = ['sr','to','lev','vol','mdd'])
    df['sr'] = pnl.resample(freq).mean()/pnl.resample(freq).std()*np.sqrt(nbar1d*255)
    df['to'] = to.resample(freq).mean() * nbar1d
    df['lev'] = lev.resample(freq).mean()
    df['vol'] = pnl.resample(freq).std()*np.sqrt(nbar1d*255)
    df['mdd'] = dd.resample(freq).max()
    print(df)

def test_corr(sgl1, sgl2, ret, start = None, end = None):
    if start is None:
        maxn = sgl1.tail(20).count(axis=1).mean() * 0.5
        c1 = sgl1[sgl1.count(axis=1) > maxn].index[0]
        maxn = sgl2.tail(20).count(axis=1).mean() * 0.5
        c2 = sgl2[sgl2.count(axis=1) > maxn].index[0]
        start = max(c1,c2)
    if end is None:
        end = min(sgl1.index[-1], sgl2.index[-1], ret.index[-1])
    return pd.DataFrame({'sgl1': (sgl1.shift() * ret).sum(axis=1), 'sgl2': (sgl2.shift() * ret).sum(axis=1) })[start:end].corr()


def signal_filters(sgl, keep_or_drop = True):
    mthend = sgl.groupby(sgl.index.to_period("M")).tail(1).index
    hmax = sgl.expanding().max()
    hmin = sgl.expanding().min()
    hmax.loc[[i for i in hmax.index if i not in mthend]] = np.nan
    hmin.loc[[i for i in hmin.index if i not in mthend]] = np.nan
    hmax = hmax.ffill()
    hmin = hmin.ffill()
    if keep_or_drop:
        return sgl.mask(sgl > hmax, hmax).mask(sgl < hmin, hmin)
    else:
        return sgl.mask(sgl > hmax, np.nan).mask(sgl < hmin, np.nan)


def get_cta_weight(signal, ret, start = None, end = None, v_lookback = 125, tarvol = 0.1/80):
    """"""
    if start is not None:
        signal = signal[start:]
    if end is not None:
        signal = signal[:end]
    cate = signal.rolling(90).rank(pct=True)
    signal = signal.mask(cate > 0.8, 2)
    signal = signal.mask(np.logical_and(cate > 0.6, cate <= 0.8), 1)
    signal = signal.mask(np.logical_and(cate > 0.4, cate <= 0.6), 0)
    signal = signal.mask(np.logical_and(cate > 0.2, cate <= 0.4), -1)
    signal = signal.mask(cate <= 0.2, -2)
    signal = signal.fillna(0) 
    z = (signal.shift() * ret.reindex_like(signal)).sum(axis=1)
    signal = signal.div(z.rolling(v_lookback).std().clip(lower = tarvol/2), axis=0) * tarvol
    return signal

def _get_rsi(pclose, ret, window):
    _mom = ret.mask(ret > 2, 0 ).mask(ret< -1, 0)
    # Upward price change
    _diff = pclose.diff()
    _upc = _diff.mask(_mom < 0, 0)
    _dpc = _diff.mask(_mom > 0, 0) * -1
    _supc = _upc.ewm(window,min_periods = window).mean()
    _sdpc = _dpc.ewm(window,min_periods = window).mean()
    _rsi = 1 - (1./((_supc/_sdpc)+1))
    _rsi = _rsi.replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return _rsi

def _get_volumeprofile(popen, phigh, plow, pclose, volume, window = 255):
    """"""
    center = (pclose + plow + phigh)/3.# * volume
    vp = pd.DataFrame().reindex_like(center).fillna(0)
    for i in range(window):
        vp += volume.shift(i).mask(pclose >= center.shift(i), 0)
    return vp / volume.rolling(window,window).sum()


def get_vp_rev(volumeprofile7, volumeprofile30, ret):
    """"""
    short = -1
    short *= (volumeprofile7 > volumeprofile30)
    short *= (volumeprofile30 > 0.5)
    short *= 1
    long = 1
    long *= (volumeprofile30 < 0.5)
    long *= (volumeprofile7 < volumeprofile30)
    long *= 1
    pos = 1 * (long + short)
    return pos[pos!=0].ffill(limit = 125).fillna(0)

def get_volhourlyseasonality_cta(ret, vwindow = 24, window = 168, swindow = 8):
    vol = ret.ewm(vwindow).std()
    mn = pd.DataFrame(index=ret.index,columns = ret.columns).fillna(0)
    sd = pd.DataFrame(index=ret.index,columns = ret.columns).fillna(0)
    for i in range(window):
        mn += vol.shift(i * 25)
        sd += vol.shift(i * 25) ** 2
    bf = (vol - mn/window)/((sd/window) - (mn/window)**2)
    bf = bf.clip(-3,3).rolling(swindow).mean()
    return bf



def get_dir_weight(signal, ret, start = None, end = None, v_lookback = 125, tarvol = None):
    """"""
    if start is not None:
        signal = signal[start:]
    if end is not None:
        signal = signal[:end]
    signal = signal.fillna(0)
    z = (signal.shift() * ret.reindex_like(signal)).sum(axis=1)
    signal = signal.div(z.rolling(v_lookback).std().clip(lower = tarvol/2), axis=0) * tarvol
    return signal


def load_data(indname):
    future = pl.read_csv(f"eq1min/{indname}_future.csv")
    future.columns = ["end_tm", "open", "high", "low", "close", "open_interest", "volume", "vwap"]
    future = future.with_columns(
                        (pl.col("end_tm").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.replace_time_zone("Asia/Hong_Kong") + pl.duration(minutes=1)).alias("end_tm"),
                        pl.col("open").cast(float).alias("open"),
                        pl.col("open_interest").cast(float).alias("open_interest"),
                        pl.col("high").cast(float).alias("high"),
                        pl.col("low").cast(float).alias("low"),
                        pl.col("close").cast(float).alias("close"),
                        pl.col("volume").cast(float).alias("volume"),
                        pl.col("vwap").cast(float).alias("vwap"),
                    )
    start_morning = time(9, 15)
    end_morning   = time(12, 0)
    start_afternoon = time(13, 0)
    end_afternoon   = time(16, 30)
    future = future.filter(
        (
            (pl.col("end_tm").dt.time() > start_morning)
            & (pl.col("end_tm").dt.time() <= end_morning)
        )
        |
        (
            (pl.col("end_tm").dt.time() > start_afternoon)
            & (pl.col("end_tm").dt.time() <= end_afternoon)
        )
    )
    spot = pl.read_csv(f"eq1min/{indname}_index.csv")
    return future

ins = 'hstech'
p1m = load_data(ins)

p1m = p1m.sort('end_tm').with_columns(
                ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return1m"),
                (pl.col("volume").diff()).alias("amt_diff"),
                (pl.col("open_interest").diff()).alias("oi_diff"),
                (pl.col("volume") * (pl.col("close") > pl.col("open")).cast(float)).alias("buy_volume"),
            ).with_columns(
                (pl.col("return1m") > pl.col("return1m").rolling_quantile(0.8, window_size = 375 * 5)).alias("topret"),
                (pl.col("return1m") < pl.col("return1m").rolling_quantile(0.2, window_size = 375 * 5)).alias("botret"),
                (pl.col("close") > pl.col("high").rolling_quantile(0.8, window_size = 375 * 5)).alias("highprice"),
                (pl.col("close") < pl.col("low").rolling_quantile(0.2, window_size = 375 * 5)).alias("lowprice"),
                (pl.col("close") > pl.col("close").rolling_mean(15)).alias("abovetwap"),
                (pl.col("close") < pl.col("close").rolling_mean(15)).alias("belowtwap"),
                (pl.col("volume") > pl.col("volume").rolling_quantile(0.8, window_size = 375 * 5)).alias("activebar"),
                (pl.col("volume") < pl.col("volume").rolling_quantile(0.2, window_size = 375 * 5)).alias("quietbar"),
                (pl.col("amt_diff") > pl.col("amt_diff").rolling_quantile(0.8, window_size = 375 * 5)).alias("incamtbar"),
                (pl.col("amt_diff") < pl.col("amt_diff").rolling_quantile(0.2, window_size = 375 * 5)).alias("decamtbar"),
                pl.col('volume').rolling_sum(window_size = 375).alias("mvolume"),
                pl.col('buy_volume').rolling_sum(window_size = 375).alias("mbuy_volume"),
                #
                pl.rolling_corr(pl.col("close"), pl.col("volume"), window_size = 60).alias("cpv"),
                pl.rolling_corr(pl.col("close"), pl.col("volume").shift(), window_size = 60).alias("cpl1v"),
                pl.rolling_corr(pl.col("close"), pl.col("buy_volume").shift(), window_size = 60).alias("cpl1bv"),
                pl.rolling_corr(pl.col("return1m"), (pl.col("high")/pl.col("low") - 1), window_size = 60).alias("crvol"),
                pl.rolling_corr(pl.col("return1m"), (pl.col("high")/pl.col("low") - 1).shift(), window_size = 60).alias("crl1vol"),
                pl.rolling_corr(pl.col("volume"), (pl.col("high")/pl.col("low") - 1), window_size = 60).alias("camtvol"),
                pl.rolling_corr(pl.col("buy_volume"), (pl.col("high")/pl.col("low") - 1), window_size = 60).alias("cbamtvol"),
                pl.rolling_corr(pl.col("return1m"), pl.col("volume"), window_size = 60).alias("crv"),
                pl.rolling_corr(pl.col("return1m"), pl.col("volume").shift(), window_size = 60).alias("crl1v"),
                pl.rolling_corr(pl.col("return1m"), pl.col("buy_volume"), window_size = 60).alias("crbv"),
            )

# amt and bins
features = p1m.group_by_dynamic(
                    index_column="end_tm",
                    every="15m",  # Set the time interval for bars (e.g., 1 minute)
                    closed="right", # Close the interval on the right
                    label = 'right'
                ).agg([
                    # sbins
                    pl.col("close").last().alias("close"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("open").first().alias("open"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("close").mean().alias("twap"),
                    ((pl.col("vwap") * pl.col("volume")).sum()/pl.col("volume").sum()).alias("vwap"),
                    ((pl.col('buy_volume')) * pl.col('highprice').cast(float)).sum().alias("bahvolume"),
                    ((pl.col('volume') - pl.col('buy_volume'))* pl.col('lowprice').cast(float)).sum().alias("salvolume"),
                    ((pl.col('volume')) * pl.col('highprice').cast(float)).sum().alias("highvolume"),
                    ((pl.col('volume')) * pl.col('lowprice').cast(float)).sum().alias("lowvolume"),
                    (pl.col('volume') * pl.col('activebar').cast(float)).sum().alias("activevolume"),
                    (pl.col('volume') * pl.col('quietbar').cast(float)).sum().alias("quietvolume"),
                    (pl.col('volume') * pl.col('topret').cast(float)).sum().alias("upmvolume"),
                    (pl.col('volume') * pl.col('botret').cast(float)).sum().alias("downmvolume"),
                    (pl.col('return1m') * pl.col('activebar').cast(float)).sum().alias("activeret"),
                    (pl.col('return1m') * pl.col('quietbar').cast(float)).sum().alias("quietret"),
                    (pl.col('return1m') * pl.col('incamtbar').cast(float)).sum().alias("incamtret"),
                    (pl.col('return1m') * pl.col('decamtbar').cast(float)).sum().alias("decamtret"),
                    (pl.col('return1m') * pl.col('highprice').cast(float)).sum().alias("highprret"),
                    (pl.col('return1m') * pl.col('lowprice').cast(float)).sum().alias("lowprret"),
                    (pl.col('activebar').cast(float)).sum().alias("activebar"),
                    (pl.col('quietbar').cast(float)).sum().alias("quietbar"),
                    (pl.col('highprice').cast(float)).sum().alias("highprice"),
                    (pl.col('lowprice').cast(float)).sum().alias("lowprice"),
                    (pl.col('incamtbar').cast(float)).sum().alias("incamtbar"),
                    (pl.col('decamtbar').cast(float)).sum().alias("decamtbar"),
                    (pl.col('topret').cast(float)).sum().alias("topret"),
                    (pl.col('botret').cast(float)).sum().alias("botret"),
                    (pl.col('abovetwap').cast(float)).sum().alias("abovetwap"),
                    (pl.col('belowtwap').cast(float)).sum().alias("belowtwap"),
                    (pl.col("volume")).skew().alias("amtskew"),
                    ((pl.col("close")/pl.col("open")-1).abs()/pl.col("volume")).mean().alias("amihudilliq"),
                    ((pl.col("high")/pl.col("low")-1)/pl.col("volume")).mean().alias("amihudilliqhl"),
                    (-pl.col("volume")/pl.col("volume").sum() * (pl.col("volume")/pl.col("volume").sum() + 1e-4).log()).sum().alias("amtenergy"),
                    # volume
                    (pl.col('return1m') / pl.col('return1m').std()).pow(3).mean().alias("skew"),
                    pl.corr(pl.col("return1m"), pl.col("return1m").shift()).alias("retauto1"),
                    (pl.col('close') > pl.col('open')).cast(int).sum().alias("upct"),
                    (pl.col('close') < pl.col('open')).cast(int).sum().alias("downct"),
                    ((pl.col('close') > pl.col('open')).cast(int) * pl.col("volume")).sum().alias("upvolume"),
                    ((pl.col('close') > pl.col('open')).cast(int) * pl.col("buy_volume")).sum().alias("upbuyvolume"),
                    ((pl.col("return1m") * pl.col('topret').cast(float))  **2).sum().alias("lpm"),
                    (pl.col("return1m").sum()/pl.col("return1m").abs().sum()).alias("smooth"),
                    (pl.col("close") / pl.col("low").min() - 1).mean().alias("pullback"),
                    (pl.col("low") <= pl.col("low").min() * 1.002).sum().alias("lbreak"),
                    (pl.col("high") >= pl.col("high").max() * 0.998).sum().alias("hbreak"),
                    (((pl.col("close") - pl.col("low"))*pl.col("volume")).sum() / ((pl.col("high") - pl.col("low"))*pl.col("volume")).sum()).alias("amtsupp"),
                    # pv rel
                    pl.col("cpv").last().alias("cpv"),
                    pl.col("cpl1v").last().alias("cpl1v"),
                    pl.col("cpl1bv").last().alias("cpl1bv"),
                    pl.col("crvol").last().alias("crvol"),
                    pl.col("crl1vol").last().alias("crl1vol"),
                    pl.col("camtvol").last().alias("camtvol"),
                    pl.col("cbamtvol").last().alias("cbamtvol"),
                    pl.col("crv").last().alias("crv"),
                    pl.col("crl1v").last().alias("crl1v"),
                    pl.col("crbv").last().alias("crbv"),
                    #pl.corr(pl.col("close"), pl.col("volume")).alias("cpv"),
                    #pl.corr(pl.col("close"), pl.col("volume").shift()).alias("cpl1v"),
                    #pl.corr(pl.col("close"), pl.col("buy_volume")).alias("cpbv"),
                    #pl.corr(pl.col("return1m"), (pl.col("high")/pl.col("low") - 1)).alias("crvol"),
                    #pl.corr(pl.col("return1m"), (pl.col("high")/pl.col("low") - 1).shift()).alias("crl1vol"),
                    #pl.corr(pl.col("volume"), (pl.col("high")/pl.col("low") - 1)).alias("camtvol"),
                    #pl.corr(pl.col("buy_volume") / pl.col("volume") - 1, (pl.col("high")/pl.col("low") - 1)).alias("cbamtvol"),
                    #pl.corr(pl.col("return1m"), pl.col("volume")).alias("crv"),
                    #pl.corr(pl.col("return1m"), pl.col("volume")).alias("crl1v"),
                    #pl.corr(pl.col("return1m"), pl.col("buy_volume")).alias("crbv"),
                    #pl.corr(pl.col("close"), pl.col("volume") - pl.col("mvolume")).alias("cpav"),
                    #pl.corr(pl.col("close"), pl.col("buy_volume") - pl.col("mbuy_volume")).alias("cpabv"),
                    #pl.corr(pl.col("return1m"), pl.col("volume") - pl.col("mvolume")).alias("crav"),
                    #pl.corr(pl.col("return1m"), (pl.col("volume") - pl.col("mvolume")).shift()).alias("crl1av"),
                    #pl.corr(pl.col("return1m"), pl.col("buy_volume") - pl.col("mbuy_volume")).alias("crabv"),
                    # vol
                    (pl.col('return1m') ** 2).sum().alias("rv"),
                    (pl.col("high")/pl.col("low") - 1).mean().alias("hlspread"),
                    ((pl.col("high") - pl.col("low"))/((pl.col("close") - pl.col("open")).abs() + 1e-5)).mean().alias("shadowsz"),
                    ((pl.col("high") - pl.col("low"))/((pl.col("close") - pl.col("open")).abs() + 1e-5)).max().alias("maxshadowsz"),
                    ((pl.col("return1m").filter(pl.col("return1m") > 0)) ** 2).sum().alias("posv"),
                    ((pl.col("return1m").filter(pl.col("return1m") < 0) ** 2).sum()).alias("negv"),
                    (pl.col("return1m").filter(pl.col("activebar")) ** 2).sum().alias("activevar"),
                    (pl.col("return1m").filter(pl.col("quietbar")) ** 2).sum().alias("quietvar"),
                    (pl.col("return1m").filter(pl.col("abovetwap")) ** 2).sum().alias("abtwapvar"),
                    (pl.col("return1m").filter(pl.col("belowtwap")) ** 2).sum().alias("bwtwapvar"),
                    (pl.col("return1m").filter(pl.col("highprice"))**2).sum().alias("highvar"),
                    (pl.col("return1m").filter(pl.col("lowprice"))**2).sum().alias("lowvar"),
                    pl.corr((pl.col("high")/pl.col("low") - 1), (pl.col("high")/pl.col("low") - 1).shift()).alias("volauto"),
                    # oi
                    #pl.corr(pl.col("close"), pl.col("open_interest")).alias("cpoi"),
                    #pl.corr(pl.col("volume"), pl.col("open_interest")).alias("caoi"),
                    #pl.corr(pl.col("return1m"), pl.col("oi_diff")).alias("cretdoi"),
                    #l.corr(pl.col("return1m"), pl.col("oi_diff").shift()).alias("cretl1doi"),
                    #l.corr(pl.col("return1m"), pl.col("open_interest")).alias("cretoi"),
                    #pl.corr(pl.col("return1m"), pl.col("open_interest").shift()).alias("cretl1oi"),
                    #((pl.col("oi_diff"))/pl.col("volume")).mean().alias("rdoiv")
                ]).with_columns(
                    ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("returns"),
                    ((pl.col("vwap") / pl.col("vwap").shift(1)) - 1).alias("vreturns"),
                    ((pl.col("twap") / pl.col("twap").shift(1)) - 1).alias("treturns"),
                )


# counts: 'activebar', 'quietbar', 'highprice', 'lowprice', 'incamtbar', 'decamtbar', 'topret', 'botret',
activebar = features.select(['end_tm', 'activebar']).to_pandas().set_index('end_tm').rename(columns = {'activebar': ins})
quietbar = features.select(['end_tm', 'quietbar']).to_pandas().set_index('end_tm').rename(columns = {'quietbar': ins})
decamtbar = features.select(['end_tm', 'decamtbar']).to_pandas().set_index('end_tm').rename(columns = {'decamtbar': ins})
incamtbar = features.select(['end_tm', 'incamtbar']).to_pandas().set_index('end_tm').rename(columns = {'incamtbar': ins})
highprice = features.select(['end_tm', 'highprice']).to_pandas().set_index('end_tm').rename(columns = {'highprice': ins})
lowprice = features.select(['end_tm', 'lowprice']).to_pandas().set_index('end_tm').rename(columns = {'lowprice': ins})
abovetwap = features.select(['end_tm', 'abovetwap']).to_pandas().set_index('end_tm').rename(columns = {'abovetwap': ins})
belowtwap = features.select(['end_tm', 'belowtwap']).to_pandas().set_index('end_tm').rename(columns = {'belowtwap': ins})
#%% test one feature
cret = features.select(['end_tm', 'returns']).to_pandas().set_index('end_tm').rename(columns = {'returns': ins}).fillna(0)
ret = features.select(['end_tm', 'treturns']).to_pandas().set_index('end_tm').rename(columns = {'treturns': ins}).fillna(0).shift(-1)
pclose = features.select(['end_tm', 'close']).to_pandas().set_index('end_tm').rename(columns = {'close': ins})
popen = features.select(['end_tm', 'open']).to_pandas().set_index('end_tm').rename(columns = {'open': ins})
phigh = features.select(['end_tm', 'high']).to_pandas().set_index('end_tm').rename(columns = {'high': ins})
plow = features.select(['end_tm', 'low']).to_pandas().set_index('end_tm').rename(columns = {'low': ins})
pvolume = features.select(['end_tm', 'volume']).to_pandas().set_index('end_tm').rename(columns = {'volume': ins})
rsi12 = _get_rsi(pclose, cret, 12)
rsi25 = _get_rsi(pclose, cret, 25)
rsi75 = _get_rsi(pclose, cret, 75)
vp25 = _get_volumeprofile(popen, phigh, plow, pclose, pvolume, window = 25)
vp125 = _get_volumeprofile(popen, phigh, plow, pclose, pvolume, window = 125)
vp500 = _get_volumeprofile(popen, phigh, plow, pclose, pvolume, window = 500)

# mom
fname = 'highprret'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'lowprret'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0) - sgln.fillna(0)).ewm(4).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
mom1 = get_dir_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)


fname = 'upct'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'downct'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0)  - sgln.fillna(0) ).ewm(25).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
mom2 = get_dir_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

#%%
fname = 'abovetwap'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'belowtwap'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0)  - sgln.fillna(0) ).ewm(12).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
mom3 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

fname = 'incamtbar'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'decamtbar'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0)  - sgln.fillna(0) ).ewm(4).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
mom4 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

cmom = (mom1.fillna(0) * 0.25 + mom2.fillna(0) * 0.25 + mom3.fillna(0) * 0.25 + mom4.fillna(0) * 0.25)

# amt
#'bahvolume', 'salvolume', 'highvolume', 'lowvolume', 'activevolume', 'quietvolume', 'upmvolume', 'downmvolume', 'upvolume', 'upbuyvolume',
# 'amtskew', 'amihudilliq', 'amihudilliqhl', 'amtenergy',
fname = 'upbuyvolume'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})#ewm(25).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
amt1 = get_dir_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

fname = 'bahvolume'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'salvolume'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'volume'
norm = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0) - sgln.fillna(0))/norm#.ewm(4).mean()
amt2 = get_cta_weight(sgl, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

fname = 'amtenergy'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins}).ewm(4).mean() * -1
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
amt3 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

camt = (amt1.fillna(0) * 0.33 + amt2.fillna(0) * 0.33 + amt3.fillna(0) * 0.33)

#%% corr
# 'cpv', 'cpl1v', 'cpl1bv', 'crvol', 'crl1vol', 'camtvol', 'cbamtvol', 'crv', 'crl1v', 'crbv',
fname = 'crl1v'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})#.ewm(25).mean() * -1
sgl = (sgl).ewm(4).mean() * -1
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
corr1 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

fname = 'cpl1bv'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})#.ewm(25).mean() * -1
sgl = (sgl).ewm(25).mean() * -1
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
corr2 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)


fname = 'crl1vol'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})#.ewm(25).mean() * -1
sgl = (sgl).ewm(25).mean() * -1
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
corr3 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

ccorr = (corr1.fillna(0) * 0.33 + corr2.fillna(0) * 0.33 + corr3.fillna(0) * 0.33)


#%% 
# 'rv', 'hlspread', 'shadowsz', 'maxshadowsz', 'posv', 'negv', 'activevar', 'quietvar', 'abtwapvar', 'bwtwapvar', 
# 'highvar', 'lowvar', 'volauto'

fname = 'smooth'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins}).ewm(4).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
fname = 'hlspread'
sgl = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins}).ewm(4).mean()
z2 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z2 = z2.fillna(0).clip(-5,5)
z = z1 - z2
vol1 = get_cta_weight(z, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

fname = 'posv'
sglp = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
fname = 'negv'
sgln = features.select(['end_tm', fname]).to_pandas().set_index('end_tm').rename(columns = {fname: ins})
sgl = (sglp.fillna(0) - sgln.fillna(0)) / (sglp.fillna(0) + sgln.fillna(0))#.ewm(4).mean()
sgl = sgl.ewm(4).mean()
z1 = (sgl - sgl.rolling(125).mean()) / sgl.rolling(125).std() 
z1 = z1.fillna(0).clip(-5,5)
vol2 = get_cta_weight(z1, ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

cvol = (vol1.fillna(0) * 0.7 + vol2.fillna(0) * 0.3)


#%% mid-freq
mid1 = get_dir_weight(get_volhourlyseasonality_cta(ret, vwindow = 25, window = 5, swindow = 12), ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)
mid2 = get_cta_weight(get_vp_rev(vp25, vp125, ret),ret, v_lookback = 500, tarvol = 0.1/80).clip(-0.2, 0.2)

cmid = (mid1.fillna(0) * 0.5 + mid2.fillna(0) * 0.5)

#%%
cta_strats = {
    'cmom': cmom,
    'camt': camt,
    'cvol': cvol,
    'ccorr': ccorr,
    'cmid': cmid
}
cta_factors = {
    'cmom': 0.2,
    'camt': 0.2,
    'cvol': 0.2,
    'ccorr': 0.,
    'cmid': 0.2
}

#chof = run_comb_mn(cta_strats, cta_factors, ret, '2019-01-01', max_w = 0.3, tarvol = 0.1/80, fee = 2e-4, 
#               dev = True, plot_path = 'eq1min/hof2.png', scale = 3)