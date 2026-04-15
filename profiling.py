import os

try:
    from line_profiler import profile as _line_profile
except ImportError:
    _line_profile = None


def _is_enabled() -> bool:
    flag = os.getenv("ENABLE_LINE_PROFILER", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def profile(func):
    """Use line_profiler only when explicitly enabled via env var.

    Default path returns the original function to avoid runtime overhead.
    """
    if _line_profile is None:
        return func
    if not _is_enabled():
        return func
    return _line_profile(func)
