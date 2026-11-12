import os
from pathlib import Path
import mlx.core as mx
import numpy as np
import pytest
from cactus.mlx.checkpoint import Checkpoint, books
from cactus.mlx.ops import CQEmbedding
from cactus.mlx.reference import decode_rows


def test_embedding_reads_indexed_rows_after_checkpoint_closed():
    path = Path(os.environ.get('CQ_MODEL_FILE', str(Path(__file__).resolve().parents[2] / 'model.safetensors')))
    if not path.exists():
        pytest.skip('Set CQ_MODEL_FILE to validate real checkpoint embeddings')
    cp = Checkpoint(path)
    name = 'model.embeddings.weight'
    embedding = CQEmbedding(cp, name, books(cp)['4-ffq'])
    ids = np.array([[0, 1, embedding._n-1], [18, 18, 512]])
    expected = decode_rows(cp,name,ids.reshape(-1)).reshape(2,3,-1)
    cp.close()
    actual = embedding.gather(mx.array(ids), dtype=mx.float32)
    np.testing.assert_allclose(np.array(actual),expected,atol=1e-7,rtol=1e-6)
    assert 'codes' not in embedding.parameters()
    with pytest.raises(IndexError):
        embedding(mx.array([embedding._n]))
