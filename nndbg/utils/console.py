"""
Shared Rich console instance for all NNDbg output.
Import this wherever you need to print something.
"""

from rich.console import Console
from rich.theme import Theme

# Custom theme for NNDbg
_theme = Theme({
    "info":     "bold cyan",
    "success":  "bold green",
    "warning":  "bold yellow",
    "error":    "bold red",
    "layer":    "dim white",
    "score":    "bold magenta",
    "axis":     "bold blue",
    "model":    "bold white",
    "header":   "bold cyan",
    "highlight":"bold yellow",
})

console = Console(theme=_theme)