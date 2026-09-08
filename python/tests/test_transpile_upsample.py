from __future__ import annotations

import torch

from cactus.transpile.capture_pytorch import capture_model


class NearestUpsample(torch.nn.Module):
    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


def _nodes(captured):
    nodes = captured.ir_graph.nodes
    return list(nodes.values()) if isinstance(nodes, dict) else list(nodes)


def _upsample_node(scale_factor: int):
    captured = capture_model(NearestUpsample(scale_factor).eval(), (torch.randn(1, 3, 5, 7),))
    matches = [n for n in _nodes(captured) if n.op == "upsample_nearest2d"]
    assert len(matches) == 1, f"expected one upsample node, got {[n.op for n in _nodes(captured)]}"
    return matches[0]


def test_scale_factor_is_recovered_from_the_exported_graph():
    assert _upsample_node(2).attrs["scale_factor"] == 2
    assert _upsample_node(3).attrs["scale_factor"] == 3


def test_output_size_form_is_recovered_from_the_input_shape():
    class ExplicitSize(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(x, size=(10, 14), mode="nearest")

    captured = capture_model(ExplicitSize().eval(), (torch.randn(1, 3, 5, 7),))
    matches = [n for n in _nodes(captured) if n.op == "upsample_nearest2d"]
    assert len(matches) == 1
    assert matches[0].attrs["scale_factor"] == 2
