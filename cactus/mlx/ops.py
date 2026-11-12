"""CQ matrix products and indexed embeddings for native MLX inference."""
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .checkpoint import RATE_BYTES

KERNELS = Path(__file__).parent / 'kernels'
_kernels = {}


def kernel(kind, rate, book):
    key = kind, rate, book.defines
    if key not in _kernels:
        header = book.defines + f'\n#define CQ_RATE {list(RATE_BYTES).index(rate)}\n'
        header += (KERNELS / 'common.metal').read_text()
        if kind == 'matvec_int':
            header += (KERNELS / 'int_dot.metal').read_text()
        names = {
            'prepare_permute': ['x', 'a', 'permutation'],
            'gather': ['codes', 'norms', 'scales', 'a', 'c', 'table', 'levels', 'rows'],
            'matvec_lut': ['permuted', 'codes', 'norms', 'scales', 'c', 'lut', 'parities', 'indices'],
            'matvec_int': ['permuted', 'codes', 'norms', 'scales', 'c', 'raw', 'indices'],
        }[kind]
        _kernels[key] = mx.fast.metal_kernel(
            name=f'cq_{kind}_{list(RATE_BYTES).index(rate)}',
            input_names=names, output_names=['out'],
            source=(KERNELS / f'{kind}.metal').read_text(), header=header,
            compile_options={'math_mode': 'safe'})
    return _kernels[key]


class CQLinear(nn.Module):
    def __init__(self, checkpoint, name, book):
        super().__init__()
        leaf = checkpoint.leaves[name]
        self._name, self._rate, self._book = name, leaf['rate'], book
        self._n, self._k = leaf['shape']
        self._scale_rows = leaf['scale_rows']
        for field in ('codes', 'norms', 'norm_scale', 'a', 'c'):
            if checkpoint.unit_scales and field in ('a', 'c'):
                self[field] = mx.full((self._k if field == 'a' else self._n,), 0x38, mx.uint8)
            else:
                self[field] = checkpoint.array(name + '.' + field)
        self.codes = self.codes.reshape(-1)
        mx.eval(self.parameters())

    def gather(self, rows, dtype=mx.bfloat16):
        shape = rows.shape
        rows = rows.reshape(-1).astype(mx.uint32)
        return kernel('gather', self._rate, self._book)(
            inputs=[self.codes, self.norms, self.norm_scale, self.a, self.c,
                    self._book.table, self._book.levels, rows],
            template=[('K', self._k), ('R', rows.size), ('SCALE_ROWS', self._scale_rows), ('OUT', dtype)],
            grid=(rows.size * (self._k // 128) * 32, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(rows.size, self._k)], output_dtypes=[dtype])[0].reshape(*shape, self._k)

    def matmul(self, x, indices=None, broadcast=False, expert_rows=None, relu2=False):
        shape = x.shape
        if shape[-1] != self._k:
            raise ValueError(f'{self._name}: expected input width {self._k}, got {shape[-1]}')
        topk = indices.shape[-1] if indices is not None else 1
        pairs = indices.size if indices is not None else x.size // self._k
        n = expert_rows or self._n
        ids = indices.reshape(-1).astype(mx.uint32) if indices is not None else mx.array([0], mx.uint32)
        slots = len(self._book.perm)
        permuted = kernel('prepare_permute', self._rate, self._book)(
            inputs=[x, self.a, self._book.forward],
            template=[('K', self._k), ('GROUPS', x.size // 128), ('SLOTS', slots)],
            grid=(x.size // 4, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(x.size * slots,)], output_dtypes=[mx.float16])[0]
        inputs = [permuted, self.codes, self.norms, self.norm_scale, self.c]
        if self._book.integer:
            kind = 'matvec_int'
            inputs.extend([self._book.raw, ids])
        else:
            kind = 'matvec_lut'
            inputs.extend([self._book.lut, self._book.parities, ids])
        threads = min(512 if self._rate == '15-ffq' else 1024, max(32, n // 32 * 32))
        out = kernel(kind, self._rate, self._book)(
            inputs=inputs,
            template=[('K', self._k), ('N', n), ('SCALE_ROWS', self._scale_rows),
                      ('EXPERTS', indices is not None), ('BROADCAST', broadcast), ('TOPK', topk),
                      ('OUT', x.dtype), ('RELU2', relu2)],
            grid=(n * 32, pairs, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(pairs, n)], output_dtypes=[x.dtype])[0]
        target = (*indices.shape, n) if indices is not None else (*shape[:-1], n)
        return out.reshape(target)

    def __call__(self, x):
        return self.matmul(x)


class CQEmbedding(CQLinear):
    """Direct row-offset index into the file mapping; only requested rows reach Metal."""
    def __init__(self, checkpoint, name, book):
        nn.Module.__init__(self)
        leaf = checkpoint.leaves[name]
        self._name, self._rate, self._book = name, leaf['rate'], book
        self._n, self._k = leaf['shape']
        self._row_bytes = self._k // 128 * RATE_BYTES[self._rate]
        self._code_rows = checkpoint.mapped_tensor(name + '.codes').reshape(self._n, self._row_bytes)
        self._norm_rows = checkpoint.mapped_tensor(name + '.norms')
        self._c_rows = checkpoint.mapped_tensor(name + '.c')
        self._scale_rows = leaf['scale_rows']
        self._scales = checkpoint.mapped_tensor(name + '.norm_scale').reshape(-1)
        self.a = checkpoint.array(name + '.a')

    def gather(self, rows, dtype=mx.bfloat16):
        shape = rows.shape
        ids = np.asarray(rows.reshape(-1).tolist(), dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self._n):
            raise IndexError('Embedding token outside vocabulary')
        codes = mx.array(self._code_rows[ids]).reshape(-1)
        norms = mx.array(self._norm_rows[ids])
        c = mx.array(self._c_rows[ids])
        scales = mx.array(self._scales[ids // self._scale_rows])
        out = kernel('gather', self._rate, self._book)(
            inputs=[codes, norms, scales, self.a, c, self._book.table, self._book.levels, mx.arange(len(ids), dtype=mx.uint32)],
            template=[('K', self._k), ('R', len(ids)), ('SCALE_ROWS', 1), ('OUT', dtype)],
            grid=(len(ids) * self._k // 4, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(len(ids), self._k)], output_dtypes=[dtype])[0]
        return out.reshape(*shape, self._k)

    def __call__(self, indices):
        return self.gather(indices)


class CQExperts(nn.Module):
    def __init__(self, up, down, num_experts):
        super().__init__()
        self.up, self.down = up, down
        self._up_rows = up._n // num_experts
        self._down_rows = down._n // num_experts

    def __call__(self, x, indices):
        h = self.up.matmul(x, indices, broadcast=True, expert_rows=self._up_rows, relu2=True)
        return self.down.matmul(h, indices, expert_rows=self._down_rows)
