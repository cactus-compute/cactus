import numpy as np
from cactus.mlx.checkpoint import fp8
from cactus.mlx.reference import hadamard


def test_e4m3_edges():
    actual = fp8(np.array([0, 128, 1, 7, 8, 56, 126, 254, 127, 255], np.uint8))
    np.testing.assert_equal(actual[:8], [0, -0., 1/512, 7/512, 1/64, 1, 448, -448])
    assert np.signbit(actual[1])
    assert np.isnan(actual[-2:]).all()


def test_hadamard_matches_definition():
    x = np.random.default_rng(9).normal(size=(5, 128)).astype(np.float32)
    h = np.array([[(-1) ** (i & j).bit_count() for j in range(128)] for i in range(128)], np.float32)
    np.testing.assert_allclose(hadamard(x), x @ h, atol=1e-5, rtol=1e-5)
