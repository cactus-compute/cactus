from types import SimpleNamespace

import numpy as np
import torch

from cactus.transpile.tdt_runtime import ParakeetTDTSelfAttention
from cactus.transpile.tdt_runtime import rel_pos_attention


def _random_inputs(seq_len, rel_len, *, heads=2, dim=16, padded=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, seq_len, heads, dim)
    tensors = [torch.randn(shape, generator=generator) for _ in range(4)]
    relative_key = torch.randn((1, rel_len, heads, dim), generator=generator)
    key_mask = torch.zeros((1, seq_len))
    key_mask[:, seq_len - padded:] = -10000.0
    return (*tensors, relative_key, key_mask)


def _dense_reference(query, key, value, rel_query, relative_key, key_mask, scale, window):
    seq_len = query.shape[1]
    center = (relative_key.shape[1] - 1) // 2
    out = torch.empty_like(query)
    for t in range(seq_len):
        for h in range(query.shape[2]):
            scores = []
            for j in range(seq_len):
                if window > 0 and abs(t - j) > window:
                    scores.append(float("-inf"))
                    continue
                content = torch.dot(query[0, t, h], key[0, j, h])
                position = torch.dot(rel_query[0, t, h], relative_key[0, center - (t - j), h])
                scores.append(float((content + position) * scale + key_mask[0, j]))
            weights = torch.softmax(torch.tensor(scores), dim=0)
            out[0, t, h] = weights @ value[0, :, h]
    return out


def test_rel_pos_attention_matches_dense_reference():
    for seq_len, window, rel_len in ((40, 6, 13), (40, 0, 79), (300, 20, 41)):
        inputs = _random_inputs(seq_len, rel_len)
        expected = _dense_reference(*inputs, 0.25, window)
        actual = rel_pos_attention(*inputs, 0.25, window)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _attention_layer(hidden=32, heads=4):
    torch.manual_seed(0)
    state_dict = {f"layer.{name}.weight": torch.randn(hidden, hidden) * 0.2
                  for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos")}
    state_dict["layer.pos_bias_u"] = torch.randn(heads, hidden // heads)
    state_dict["layer.pos_bias_v"] = torch.randn(heads, hidden // heads)
    config = SimpleNamespace(
        hidden_dim=hidden,
        attention_heads=heads,
        attention_head_dim=hidden // heads,
        attention_scale=(hidden // heads) ** -0.5,
    )
    return ParakeetTDTSelfAttention(config, "layer", state_dict).eval()


def test_local_attention_exports_single_semantic_op():
    from cactus.transpile.capture_pytorch import capture_model

    class _Windowed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _attention_layer()

        def forward(self, x, mask):
            return self.attention(x, mask, 4)

    captured = capture_model(_Windowed().eval(), (torch.randn(1, 12, 32), torch.zeros(1, 1, 1, 12)), strict=False)
    nodes = [
        captured.ir_graph.nodes[node_id]
        for node_id in captured.ir_graph.order
        if captured.ir_graph.nodes[node_id].op == "rel_pos_attention"
    ]
    assert len(nodes) == 1
    assert nodes[0].attrs == {"scale": 8 ** -0.5, "window_size": 4}
    assert len(nodes[0].inputs) == 6


def test_engine_op_matches_torch_reference():
    from cactus.bindings.cactus import Graph

    for seq_len, window, rel_len in ((150, 20, 41), (70, 0, 139)):
        inputs = _random_inputs(seq_len, rel_len, dim=32, padded=5, seed=1)
        inputs = tuple(tensor.half().float() for tensor in inputs)
        expected = rel_pos_attention(*inputs, 32 ** -0.5, window)

        graph = Graph()
        nodes = [graph.input(tuple(tensor.shape)) for tensor in inputs]
        out = graph.rel_pos_attention(*nodes, 32 ** -0.5, window)
        for node, tensor in zip(nodes, inputs):
            graph.set_input(node, tensor.numpy().astype(np.float16))
        graph.execute()
        np.testing.assert_allclose(out.numpy().astype(np.float32), expected.numpy(), atol=3e-3)


def test_chunked_subsampling_matches_single_pass():
    from cactus.transpile.tdt_runtime import ParakeetTDTPreEncode

    torch.manual_seed(0)
    channels, mels, hidden = 4, 16, 8
    state_dict = {
        "encoder.pre_encode.conv.0.weight": torch.randn(channels, 1, 3, 3),
        "encoder.pre_encode.conv.0.bias": torch.randn(channels),
        "encoder.pre_encode.out.weight": torch.randn(hidden, channels * 2),
        "encoder.pre_encode.out.bias": torch.randn(hidden),
    }
    for index in (2, 5):
        state_dict[f"encoder.pre_encode.conv.{index}.weight"] = torch.randn(channels, 1, 3, 3)
        state_dict[f"encoder.pre_encode.conv.{index}.bias"] = torch.randn(channels)
        state_dict[f"encoder.pre_encode.conv.{index + 1}.weight"] = torch.randn(channels, channels, 1, 1)
        state_dict[f"encoder.pre_encode.conv.{index + 1}.bias"] = torch.randn(channels)
    state_dict = {name: tensor * 0.3 for name, tensor in state_dict.items()}
    config = SimpleNamespace(subsampling_conv_channels=channels, num_mel_bins=mels, hidden_dim=hidden)
    pre_encode = ParakeetTDTPreEncode(config, state_dict).eval()
    features = torch.randn(1, 7003, mels)
    with torch.no_grad():
        chunked = pre_encode(features)
        single = pre_encode._subsample(features)
    assert chunked.shape == single.shape
    torch.testing.assert_close(chunked, single, atol=1e-5, rtol=1e-5)
