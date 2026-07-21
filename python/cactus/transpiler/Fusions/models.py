from dataclasses import dataclass
from typing import Any




@dataclass(slots=True)
class SynthNode:
    attrs: dict[str, Any]

@dataclass(slots=True)
class Route:
    pass



@dataclass(slots=True)
class SynthGraph:
    source_node: SynthNode
    route: Route