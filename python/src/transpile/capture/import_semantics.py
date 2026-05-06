from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.transpile.capture.graph_ir import IRGraph
from src.transpile.capture.graph_ir import IRNode
from src.transpile.capture.graph_ir import verify_ir

def apply_import_semantics(graph: IRGraph) -> IRGraph:
    return graph
