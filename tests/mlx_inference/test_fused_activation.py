import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cactus.mlx.checkpoint import Checkpoint, books
from cactus.mlx.ops import CQLinear


def test_fused_relu2_preserves_output_rounding_for_every_rate():
    path = Path(os.environ.get('CQ_MODEL_FILE', str(Path(__file__).resolve().parents[2] / 'model.safetensors')))
    if not path.exists():
        pytest.skip('Set CQ_MODEL_FILE to validate the packed kernels')
    cp = Checkpoint(path)
    try:
        codebooks = books(cp)
        mx.random.seed(29)
        for rate in codebooks:
            name = min((n for n, leaf in cp.leaves.items() if leaf['rate'] == rate),
                       key=lambda n: np.prod(cp.leaves[n]['shape']))
            op = CQLinear(cp, name, codebooks[rate])
            for dtype in (mx.bfloat16, mx.float16, mx.float32):
                x = mx.random.normal((1,3,op._k)).astype(dtype)
                expected = nn.relu2(op(x))
                actual = op.matmul(x, relu2=True)
                np.testing.assert_array_equal(actual.astype(mx.float32), expected.astype(mx.float32))
    finally:
        cp.close()
