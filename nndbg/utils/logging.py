"""
Rich-based logging for NNDbg.
Silent by default — no terminal output unless verbose mode is enabled.
Call nndbg.set_verbose(True) to turn on logging.
"""

import logging
from rich.logging import RichHandler
from nndbg.utils.console import console

# Global verbose flag — OFF by default
_VERBOSE: bool = False


def set_verbose(enabled: bool) -> None:
    """
    Enable or disable all NNDbg terminal output.

    Args:
        enabled: True = show INFO logs + tqdm bars.
                 False (default) = completely silent.

    Example:
        import nndbg
        nndbg.set_verbose(True)
    """
    global _VERBOSE
    _VERBOSE = enabled
    # Update all existing nndbg loggers
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("nndbg") and isinstance(logger, logging.Logger):
            logger.setLevel(logging.INFO if enabled else logging.WARNING)


def is_verbose() -> bool:
    """Return True if verbose mode is on."""
    return _VERBOSE


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
            keywords=[
                "Discovered", "Attached", "Processing", "Training",
                "Analysis complete", "Created run", "ready",
                "hooks", "probe", "layer", "axis",
            ],
        )
        logger.addHandler(handler)
        # Silent by default; respects whatever _VERBOSE is at call time
        logger.setLevel(logging.INFO if _VERBOSE else logging.WARNING)
        logger.propagate = False
    return logger