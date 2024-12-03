from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary

class SerializableLock:
    """A Serializable per-process Lock

    This wraps a normal ``threading.Lock`` object and satisfies the same
    interface.  However, this lock can also be serialized and sent to different
    processes.  It will not block concurrent operations between processes (for
    this you should look at ``dask.multiprocessing.Lock`` or ``locket.lock_file``
    but will consistently deserialize into the same lock.

    So if we make a lock in one process::

        lock = SerializableLock()

    And then send it over to another process multiple times::

        bytes = pickle.dumps(lock)
        a = pickle.loads(bytes)
        b = pickle.loads(bytes)

    Then the deserialized objects will operate as though they were the same
    lock, and collide as appropriate.

    This is useful for consistently protecting resources on a per-process
    level.

    The creation of locks is itself not threadsafe.
    """
    _locks: ClassVar[WeakValueDictionary[Hashable, threading.Lock]] = WeakValueDictionary()
    token: Hashable
    lock: threading.Lock

    def __init__(self, token: Hashable | None=None):
        self.token = token or str(uuid.uuid4())
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = threading.Lock()
            SerializableLock._locks[self.token] = self.lock

    def __enter__(self):
        self.lock.__enter__()

    def __exit__(self, *args):
        self.lock.__exit__(*args)

    def __getstate__(self):
        return self.token

    def __setstate__(self, token):
        self.__init__(token)

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.token}>'
    __repr__ = __str__
HDF5_LOCK = SerializableLock()
NETCDFC_LOCK = SerializableLock()
_FILE_LOCKS: MutableMapping[Any, threading.Lock] = weakref.WeakValueDictionary()

def _get_lock_maker(scheduler=None):
    """Returns an appropriate function for creating resource locks.

    Parameters
    ----------
    scheduler : str or None
        Dask scheduler being used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """
    if scheduler is None:
        return threading.Lock
    elif scheduler == "sync":
        return threading.Lock
    elif scheduler == "threads":
        return threading.Lock
    elif scheduler in ("processes", "multiprocessing"):
        return multiprocessing.Lock
    else:
        from dask.utils import get_scheduler_lock
        return get_scheduler_lock(scheduler)

def _get_scheduler(get=None, collection=None) -> str | None:
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See Also
    --------
    dask.base.get_scheduler
    """
    try:
        from dask.base import get_scheduler
        return get_scheduler(get, collection)
    except ImportError:
        return None

def get_write_lock(key):
    """Get a scheduler appropriate lock for writing to the given resource.

    Parameters
    ----------
    key : str
        Name of the resource for which to acquire a lock. Typically a filename.

    Returns
    -------
    Lock object that can be used like a threading.Lock object.
    """
    if key not in _FILE_LOCKS:
        scheduler = _get_scheduler()
        lock_maker = _get_lock_maker(scheduler)
        _FILE_LOCKS[key] = lock_maker()
    return _FILE_LOCKS[key]

def acquire(lock, blocking=True):
    """Acquire a lock, possibly in a non-blocking fashion.

    Includes backwards compatibility hacks for old versions of Python, dask
    and dask-distributed.
    """
    try:
        return lock.acquire(blocking=blocking)
    except TypeError:
        # Some lock objects (e.g., multiprocessing.Lock) don't support
        # the blocking keyword argument.
        if blocking:
            return lock.acquire()
        else:
            if not lock.acquire(False):
                return False
            lock.release()
            return True

class CombinedLock:
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.
    """

    def __init__(self, locks):
        self.locks = tuple(set(locks))

    def __enter__(self):
        for lock in self.locks:
            lock.__enter__()

    def __exit__(self, *args):
        for lock in self.locks:
            lock.__exit__(*args)

    def __repr__(self):
        return f'CombinedLock({list(self.locks)!r})'

class DummyLock:
    """DummyLock provides the lock API without any actual locking."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def combine_locks(locks):
    """Combine a sequence of locks into a single lock."""
    locks = [ensure_lock(lock) for lock in locks]
    if len(locks) == 0:
        return DummyLock()
    elif len(locks) == 1:
        return locks[0]
    else:
        return CombinedLock(locks)

def ensure_lock(lock):
    """Ensure that the given object is a lock."""
    if lock is None:
        return DummyLock()
    elif isinstance(lock, (threading.Lock, multiprocessing.Lock, CombinedLock, DummyLock)):
        return lock
    elif hasattr(lock, '__enter__') and hasattr(lock, '__exit__'):
        return lock
    else:
        raise TypeError(f"Object {lock} is not a valid lock")
