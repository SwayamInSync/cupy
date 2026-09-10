from __future__ import annotations

from cupy.cuda import Device, runtime


def _get_device_id(device):
    """Normalizes a non-``None`` ``device=`` argument to an integer id."""
    if isinstance(device, Device):
        return device.id
    # bool is an int subclass; reject it so True/False aren't devices 1/0.
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    raise TypeError(
        'device must be an int or cupy.cuda.Device, got '
        f'{type(device).__name__!r}')


def _on_device(device, func, *args, **kwargs):
    """Calls ``func(*args, **kwargs)`` on ``device``, then restores the device.

    Creation functions take this branch only when ``device is not None`` and
    re-enter themselves with the argument dropped, so the default path costs
    one ``is not None`` test. Passing ``func`` and its arguments rather than a
    closure keeps the callers' locals out of cells, which would otherwise slow
    down the ``device=None`` path as well.

    Calls cudart directly and keeps no state, following CuPy's convention of
    not wrapping internal device switches in a context manager.
    """
    dev = _get_device_id(device)
    prev = runtime.getDevice()
    if dev != prev:
        runtime.setDevice(dev)
    try:
        return func(*args, **kwargs)
    finally:
        if dev != prev:
            runtime.setDevice(prev)
