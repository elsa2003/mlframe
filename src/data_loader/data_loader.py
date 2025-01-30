from __future__ import annotations

import os
import glob
import yaml
import re
import pathlib
import polars as pl
from typing import Iterable

from tardis_dev import datasets, get_exchange_details
import datetime as dt
import pandas as pd
import numpy as np
import time


from utils.types import (
    DatetimeType,
    DateType,
    FreqType,
    PathType,
    Generator
)

from utils.utils import (
    genenerate_dates,
    to_date
)

_cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_cwd, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)




def get_hist_spot_trades_data(
        symbol: str,
        source: str,
        #period: FreqType,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:
    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'trades',
        exchange = 'binance',
    )
    if source == 'tardis':
        save_path = os.path.join(config['data']['data_dir'], f"binance/trades/")
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'side', 'price', 'amount']
                )
        )
    return tdata



def get_hist_perp_trades_data(
        symbol: str,
        source: str,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:
    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'trades',
        exchange = 'binance-futures',
    )
    if source == 'tardis':
        save_path = os.path.join(config['data']['data_dir'], f"binance-futures/trades/")
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'side', 'price', 'amount']
                )
        )
    return tdata



def get_hist_perp_tickers_data(
        symbol: str,
        source: str,
        start_date: DateType,
        end_date: DateType,
    ) -> pl.DataFrame:

    assert source in ('tardis',)
    #assert period in ('5m',) # currently only support '5m\
    _download_data_from_tardis(
        sdate = start_date, edate = end_date,
        symbol = symbol,
        data_type = 'derivative_ticker',
        exchange = 'binance-futures',
    )
    if source == 'tardis':
        save_path = os.path.join(config['data']['data_dir'], f"binance-futures/derivative_ticker/")
        tdata = (
            _read_csv(dir_path=save_path,sdate=start_date,edate=end_date).with_columns((pl.col("timestamp")).cast(pl.Datetime).alias("datetime"))
                .select(
                    ["datetime", "symbol", 'open_interest', 'last_price', 'index_price', 'mark_price']
                )
        )
    return tdata




def _download_data_from_tardis(
    sdate: str, edate: str,
    symbol: str,
    data_type: str,
    exchange: str,
    ):
    symbol = symbol.lower()
    save_path = os.path.join(config['data']['data_dir'], f"{exchange}/{data_type}/")
    dates = _get_tardis_dir_missing_dates(save_path, sdate, edate)
    try:
        details = get_exchange_details(exchange)
    except RuntimeError:
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        details = get_exchange_details(exchange)

    t = pd.DataFrame(details["availableSymbols"])
    for date in dates:
        std1, edd1 = _generate_one_date_range(date)
        since_date = str(max(dt.date.fromisoformat(t[t['id']==symbol]['availableSince'].values[0].split('T')[0]), dt.date.fromisoformat(std1)))
        if t[t['id']==symbol]['availableTo'].values[0] is np.nan:
            to_date =edd1
        else:
            to_date = str(min(dt.date.fromisoformat(t[t['id']==symbol]['availableTo'].values[0].split('T')[0]), dt.date.fromisoformat(edd1)))
        datasets.download(
            exchange=exchange,
            # data_types=[ "trades", "quotes", "derivative_ticker", "book_snapshot_25", "book_snapshot_5", "liquidations"],
            data_types=[data_type],
            from_date=since_date,
            to_date=to_date,
            symbols=[symbol],
            api_key=config['tardis']['key'],
            download_dir=save_path,
        )
        time.sleep(1)

def _generate_one_date_range(date: DateType):
    return (date.strftime('%Y-%m-%d'),(date + dt.timedelta(days=1)).strftime('%Y-%m-%d'))


def _get_tardis_dir_missing_dates(wkdir:str, sdate:str, edate:str):
    fs = glob.glob(os.path.join(wkdir, '*.csv.gz'))
    existing_dates = []
    for f in fs:
        date = _extract_date(f)
        if date is not None:
            existing_dates.append(date)
    dates = genenerate_dates(sdate, edate)
    dates = [i for i in dates if i.strftime('%Y-%m-%d') not in existing_dates]
    return dates


def _extract_date(filename):
    # Define the regex pattern to match the date in the format YYYY-MM-DD
    pattern = r'\d{4}-\d{2}-\d{2}'
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    # If a match is found, return the matched string
    if match:
        return match.group(0)
    else:
        return None


def _extract_files_by_date(
        dir_path: PathType,
        suffix: str,
        sdate: DateType | None = None,
        edate: DateType | None = None,
        filename_pattern: str | re.Pattern | None = None,
    ) -> Generator[pathlib.Path]:
    dir_path = pathlib.Path(dir_path)
    sdate = to_date(sdate) if sdate is not None else dt.date(2020, 1, 1)
    edate = to_date(edate) if edate is not None else dt.date(2300, 12, 31)
    if not filename_pattern:
        files = (
            fl
            for fl in dir_path.glob(f"*{suffix}")
            if sdate <= dt.date.fromisoformat(_extract_date(fl.stem)) <= edate
        )
    else:
        filename_pattern = re.compile(filename_pattern)
        files = (
            fl
            for fl in dir_path.glob(f"*{suffix}")
            for extracted_date_list in [re.findall(filename_pattern, fl.stem)]
            if extracted_date_list
            if sdate <= to_date(extracted_date_list[0]) <= edate
        )
    return files


def _read_csv(    
        dir_path: PathType,
        sdate: DateType | None = None,
        edate: DateType | None = None,
        time: TimeType | None = None,
        columns: list[str] | None = None,
        sort_cols: list[str] | None = None,
        filename_pattern: str | re.Pattern | None = None,
    ) -> pl.DataFrame:
    _files = _extract_files_by_date(
        dir_path=dir_path,
        suffix=".csv.gz",
        sdate=sdate,
        edate=edate,
        filename_pattern=filename_pattern,
    )
    _files_list = sorted(_files)
    res = pl.concat(
                    [
                        pl.read_csv(source=f, columns=columns).shrink_to_fit() 
                        for f in _files_list
                    ],
                    how="diagonal",
                )
    return res

