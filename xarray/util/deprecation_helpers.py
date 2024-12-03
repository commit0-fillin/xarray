import inspect
import warnings
from functools import wraps
from typing import Callable, TypeVar
from xarray.core.utils import emit_user_level_warning
T = TypeVar('T', bound=Callable)
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
EMPTY = inspect.Parameter.empty

def _deprecate_positional_args(version) -> Callable[[T], T]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    ``*`` will issue a warning when passed as a positional argument.

    Parameters
    ----------
    version : str
        version of the library when the positional arguments were deprecated

    Examples
    --------
    Deprecate passing `b` as positional argument:

    def func(a, b=1):
        pass

    @_deprecate_positional_args("v0.1.0")
    def func(a, *, b=2):
        pass

    func(1, 2)

    Notes
    -----
    This function is adapted from scikit-learn under the terms of its license. See
    licences/SCIKIT_LEARN_LICENSE
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            # Check if any keyword-only argument is passed as positional
            for name, param in sig.parameters.items():
                if param.kind == param.KEYWORD_ONLY and name in bound_args.arguments:
                    warnings.warn(
                        f"Passing {name} as positional argument is deprecated "
                        f"since version {version} and will be removed in a future version. "
                        f"Please use keyword argument instead.",
                        FutureWarning,
                        stacklevel=2
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def deprecate_dims(func: T, old_name='dims') -> T:
    """
    For functions that previously took `dims` as a kwarg, and have now transitioned to
    `dim`. This decorator will issue a warning if `dims` is passed while forwarding it
    to `dim`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if old_name in kwargs:
            warnings.warn(
                f"The '{old_name}' argument is deprecated and will be removed in a future version. "
                f"Please use 'dim' instead.",
                FutureWarning,
                stacklevel=2
            )
            kwargs['dim'] = kwargs.pop(old_name)
        return func(*args, **kwargs)
    return wrapper
