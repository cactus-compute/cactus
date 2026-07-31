from __future__ import annotations

from typing import Any

from ..IR import models as IRModels


def element_count(shape: list[Any]) -> int | None:
    product = 1

    for dim in shape:
        if not isinstance(dim, int):
            return None

        product *= dim

    return product


def tensor_shape(node: IRModels.Node) -> list[Any]:
    return list(IRModels.tensor_shape(node))


def tensor_rank(node: IRModels.Node) -> int:
    return len(tensor_shape(node))


def tensor_last_dim(node: IRModels.Node) -> Any | None:
    shape = tensor_shape(node)
    return shape[-1] if shape else None
