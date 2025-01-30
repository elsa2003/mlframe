
import polars as pl
from utils.types import List


def generate_fret_from_bars(
    data: pl.DataFrame,
    dtname: str,
    pxname: str,
    horizons: List[int]
    ) -> pl.DataFrame:
    return data.sort(dtname).with_columns(
            ((pl.col(pxname).shift(-j)-pl.col(pxname))/pl.col(pxname)).alias(f"ret_T{j}") for j in horizons
        )

def generate_fret_cat_from_bars():
    pass


def generate_sep_fret_from_bars(
    data: pl.DataFrame,
    dtname: str,
    pxname: str,
    horizons: List[int]
    ) -> pl.DataFrame:
    return data.sort(dtname).with_columns(
            ((pl.col(pxname).shift(-j)-pl.col(pxname).shift(-i))/pl.col(pxname).shift(-i)).alias(f"ret_T{i}_{j}") for i,j in zip([0]+horizons,horizons) 
        )