from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version

def count_not_none(*args) -> int:
    """Compute the number of non-None arguments.

    Copied from pandas.core.common.count_not_none (not part of the public API)
    """
    return sum(arg is not None for arg in args)

class _NoDefault(Enum):
    """Used by pandas to specify a default value for a deprecated argument.
    Copied from pandas._libs.lib._NoDefault.

    See also:
    - pandas-dev/pandas#30788
    - pandas-dev/pandas#40684
    - pandas-dev/pandas#40715
    - pandas-dev/pandas#47045
    """
    no_default = 'NO_DEFAULT'

    def __repr__(self) -> str:
        return '<no_default>'
no_default = _NoDefault.no_default
NoDefault = Literal[_NoDefault.no_default]

def nanosecond_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
    """Return a nanosecond-precision Timestamp object.

    Note this function should no longer be needed after addressing GitHub issue
    #7493.
    """
    ts = pd.Timestamp(*args, **kwargs)
    return ts.as_unit('ns')
