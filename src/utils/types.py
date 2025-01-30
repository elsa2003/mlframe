from __future__ import annotations

import sys
import datetime as dt
import pathlib
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Iterable,
    List,
    Dict,
    Mapping,
    Sequence,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    Generator
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

DateType: TypeAlias = Union[str, dt.date]
DatetimeType: TypeAlias = Union[str, dt.datetime]
TimeType: TypeAlias = Union[str, dt.time]
PathType: TypeAlias = Union[str, pathlib.Path]
FreqType: TypeAlias = Literal["5min", "15min", "1h", "8h", "1d"]
ReturnsType: TypeAlias = Literal[
    "ret"
]
ModeType: TypeAlias = Literal["dev", "prod"]


#
class ModelType(Protocol):
    def fit(self, X, y, *args, **kwargs) -> None:
        ...

ParamType: TypeAlias = Dict[str, Any]