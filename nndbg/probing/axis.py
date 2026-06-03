from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Axis:
    name: str
    groups: Dict[str, List[str]]

    def __post_init__(self):
        if len(self.groups) < 2:
            raise ValueError(
                f"Axis '{self.name}' needs at least 2 groups, "
                f"got {len(self.groups)}: {list(self.groups.keys())}"
            )
        
        for group, samples in self.groups.items():
            if not samples:
                raise ValueError(
                    f"Group '{group}' in axis '[self.name]' has no samples"
                )
        
    @property
    def group_names(self) -> List[str]:
        return list(self.groups.keys())
    
    @property
    def total_samples(self) -> int:
        return sum(len(s) for s in self.groups.values())
    
    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{g}({len(s)})" for g, s in self.groups.items()
        )
        return f"Axis(name='{self.name}', groups=[{group_info}])"