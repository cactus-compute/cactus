from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest
from PIL import Image

from cactus.mlx.vision import ImageProcessor, resize_bicubic
from cactus.mlx.models.nemotron_vision import pixel_shuffle, resize_positions
from cactus.mlx.models.nemotron_speculative import MTPDecoder
from cactus.mlx.models.nemotron import CQModel


def test_image_expansion_keeps_image_order_and_token_counts():
    processor = ImageProcessor()
    text, pixels = processor('first <image> then <image>', images=[
        Image.new('RGB', (512, 512), 'red'), Image.new('RGB', (512, 512), 'blue')])
    assert text == 'first <img>' + '<image>' * 256 + '</img> then <img>' + '<image>' * 256 + '</img>'
    assert pixels[0].shape == pixels[1].shape == (3, 512, 512)
    assert pixels[0][0, 0, 0] > pixels[1][0, 0, 0]
    assert pixels[0][2, 0, 0] < pixels[1][2, 0, 0]
    with pytest.raises(ValueError, match='placeholders'):
        processor('<image>', images=[])


def test_embedding_merge_preserves_text_and_replaces_only_image_rows():
    model = SimpleNamespace(backbone=SimpleNamespace(embeddings=lambda ids: mx.repeat(ids[:, None], 4, axis=1)),
                            _vision=lambda pixels: mx.full((2, 4), 99))
    merged = CQModel.embed_inputs(model, mx.array([1, 18, 18, 2]), pixel_values=[np.zeros((3, 32, 64))])
    np.testing.assert_array_equal(merged, [[1]*4, [99]*4, [99]*4, [2]*4])
    with pytest.raises(ValueError, match='token count'):
        CQModel.embed_inputs(model, mx.array([1, 18, 2]), pixel_values=[np.zeros((3, 32, 64))])


def bicubic_weight(x):
    if x < 1:
        return (1.5 * x - 2.5) * x * x + 1
    if x < 2:
        return ((-0.5 * x + 2.5) * x - 4) * x + 2
    return 0


def bicubic_matrix(original, output):
    scale = np.float32(original / output)
    stretch = max(scale, np.float32(1))
    matrix = np.zeros((output, original))
    for index in range(output):
        center = scale * np.float32(index + 0.5)
        for source in range(max(0, int(center - 2 * stretch + 0.5)), min(original, int(center + 2 * stretch + 0.5))):
            matrix[index, source] = bicubic_weight(float(abs(np.float32(source + 0.5) - center) / stretch))
    return matrix / matrix.sum(axis=1, keepdims=True)


def test_resize_preserves_constant_and_float_overshoot():
    unnormalized = (0, 0, 0), (1, 1, 1)
    constant = np.full((7, 11, 3), 123, np.uint8)
    np.testing.assert_allclose(resize_bicubic(constant, 23, 17, *unnormalized), 123 / 255, atol=1e-6)
    edge = np.zeros((4, 4, 3), np.uint8)
    edge[:, 2:] = 255
    resized = resize_bicubic(edge, 16, 16, *unnormalized)
    assert resized.min() < 0 and resized.max() > 1


def test_resize_matches_pytorch_filter():
    image = np.random.default_rng(0).integers(0, 256, (513, 1025, 3), dtype=np.uint8)
    processor = ImageProcessor()
    resized = np.einsum('yh,hwc,xw->cyx', bicubic_matrix(513, 224), image, bicubic_matrix(1025, 448), optimize=True)
    expected = (resized / 255 - processor.mean[:, None, None]) / processor.std[:, None, None]
    np.testing.assert_allclose(resize_bicubic(image, 224, 448, processor.mean, processor.std), expected, atol=5e-6)


def test_patch_grids_support_rectangular_and_tiny_images():
    processor = ImageProcessor()
    assert processor.target_size(512, 512) == (512, 512)
    assert processor.target_size(256, 256) == (512, 512)
    for width, height in [(1, 1), (1, 1000), (1920, 1080), (1080, 1920), (4096, 4096)]:
        h, w = processor.target_size(width, height)
        assert h > 0 and w > 0 and h % 32 == w % 32 == 0
        assert h * w // 256 <= processor.max_patches


def test_projector_patch_channel_order():
    x = mx.arange(8).reshape(1, 8, 1)
    np.testing.assert_array_equal(pixel_shuffle(x, 2, 4), [[[0, 1, 4, 5], [2, 3, 6, 7]]])


def test_position_interpolation_uses_top_left_crop():
    positions = mx.arange(16).reshape(1, 16, 1).astype(mx.float32)
    np.testing.assert_array_equal(resize_positions(positions, 2, 4), np.arange(8).reshape(1, 8, 1))
    np.testing.assert_allclose(resize_positions(positions, 2, 2), [[[2.5], [4.5], [10.5], [12.5]]])


def test_mtp_prefill_uses_shifted_visual_embeddings():
    target_inputs, draft_inputs = [], []
    embeds = mx.arange(20).reshape(5, 4).astype(mx.float32)
    def forward(tokens, cache, input_embeddings=None):
        target_inputs.append(input_embeddings)
        return mx.zeros((1, tokens.shape[1], 8)), input_embeddings + 100
    def head(embeddings, hidden, cache):
        draft_inputs.append((embeddings, hidden))
    decoder = MTPDecoder.__new__(MTPDecoder)
    decoder.model = SimpleNamespace(args=SimpleNamespace(hidden_size=4), forward_hidden=forward)
    decoder.head, decoder.cache, decoder.draft_cache = SimpleNamespace(cache_inputs=head), [], SimpleNamespace(state=[])
    decoder.prefill([1, 18, 18, 18, 2], step=2, input_embeddings=embeds)
    np.testing.assert_array_equal(mx.concatenate(target_inputs, axis=1)[0], embeds)
    np.testing.assert_array_equal(mx.concatenate([x for x, _ in draft_inputs], axis=1)[0], embeds[1:])
    np.testing.assert_array_equal(mx.concatenate([h for _, h in draft_inputs], axis=1)[0], embeds[:-1] + 100)
