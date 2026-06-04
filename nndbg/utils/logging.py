"""
Rich-based logging for NNDbg.
Replaces plain StreamHandler with Rich's beautiful handler.
"""

import logging
from rich.logging import RichHandler
from rich.console import Console
from nndbg.utils.console import console


def get_logger(name: str) -> logging.Logger:
    # Strip nndbg. prefix for cleaner display
    short_name = name.replace("nndbg.", "")

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,        # hide file path
            markup=True,            # allow Rich markup in messages
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
            keywords=[              # these words get highlighted
                "Discovered",
                "Attached",
                "Processing",
                "Training",
                "Analysis complete",
                "Created run",
                "ready",
                "hooks",
                "probe",
                "layer",
                "axis",
            ],
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger