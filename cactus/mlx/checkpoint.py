"""Read selected tensors without materializing the full safetensors container."""
import json
import mmap
import struct
from functools import cached_property
from pathlib import Path

import mlx.core as mx
import numpy as np

DTYPES = {'U8': 'u1', 'I8': 'i1', 'U16': '<u2', 'I16': '<i2',
          'I32': '<i4', 'I64': '<i8', 'F16': '<f2', 'BF16': '<u2',
          'F32': '<f4', 'F64': '<f8'}
RATE_BYTES = {'1-ffq': 16, '15-ffq': 24, '2-ffq': 32, '4-ffq': 64}


def fp8(values):
    """E4M3FN bytes to FP32, including subnormals and signed zero."""
    bits = np.asarray(values, dtype=np.uint8)
    exp, mant = (bits >> 3) & 15, bits & 7
    v = np.where(exp == 0, mant.astype(np.float32) / 512,
                 (1 + mant.astype(np.float32) / 8) * np.exp2(exp.astype(np.int32) - 7))
    v = np.where((exp == 15) & (mant == 7), np.nan, v)
    return np.where(bits & 128, -v, v).astype(np.float32)


class Checkpoint:
    def __init__(self, path, component='language_model'):
        self.path = Path(path)
        self._file = self.path.open('rb')
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        size = struct.unpack('<Q', self._map[:8])[0]
        self.header = json.loads(self._map[8:8 + size])
        self.offset = size + 8
        meta = self.header['__metadata__']
        if meta.get('format') != 'cq-consolidated-v1':
            raise ValueError('Expected the cq-consolidated-v1 single-file checkpoint')
        self.manifest = json.loads(meta['cq.' + component])
        self.unit_scales = component == 'vision'
        if self.unit_scales:
            self.manifest = self.manifest['packed']
            if self.manifest['format'] != 'vision-packed-v1':
                raise ValueError('Unsupported packed vision format')
            self.leaves = {name: dict(leaf, rate=self.manifest['rate'], scale_rows=leaf['shape'][0])
                           for name, leaf in self.manifest['leaves'].items()}
        else:
            self.leaves = self.manifest['leaves']

    def numpy(self, name):
        h = self.header[name]
        return np.ndarray(h['shape'], dtype=DTYPES[h['dtype']], buffer=self._map,
                          offset=self.offset + h['data_offsets'][0])

    def array(self, name):
        a = self.numpy(name)
        if self.header[name]['dtype'] == 'F64':
            a = a.astype(np.float32)
        result = mx.array(a)
        return result.view(mx.bfloat16) if self.header[name]['dtype'] == 'BF16' else result

    def mapped_tensor(self, name):
        """An independent, page-aligned view; does not retain the whole checkpoint."""
        h = self.header[name]
        start, end = h['data_offsets']
        absolute = self.offset + start
        aligned = absolute // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        mapping = mmap.mmap(self._file.fileno(), absolute - aligned + end - start,
                            access=mmap.ACCESS_READ, offset=aligned)
        return np.ndarray(h['shape'], dtype=DTYPES[h['dtype']], buffer=mapping,
                          offset=absolute - aligned)

    def close(self):
        self._map.close()
        self._file.close()

    def rows(self, name, row_ids):
        leaf = self.leaves[name]
        n, k = leaf['shape']
        ids = np.asarray(row_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= n):
            raise IndexError('CQ row outside matrix')
        width = k // 128 * RATE_BYTES[leaf['rate']]
        return dict(
            codes=np.array(self.numpy(name + '.codes').reshape(n, width)[ids]),
            norms=fp8(self.numpy(name + '.norms')[ids]),
            norm_scale=np.array(self.numpy(name + '.norm_scale')).reshape(-1)[ids // leaf['scale_rows']].astype(np.float32),
            a=np.ones(k, np.float32) if self.unit_scales else fp8(self.numpy(name + '.a')),
            c=np.ones(len(ids), np.float32) if self.unit_scales else fp8(self.numpy(name + '.c')[ids]),
        )


class Book:
    def __init__(self, checkpoint, prefix):
        get = lambda k: np.array(checkpoint.numpy(prefix + '.' + k))
        self.perm, self.pat, self.lo = get('perm'), get('pat'), get('lo')
        self.q1, self.l1, self.q2, self.l2 = get('Q1'), get('l1'), get('Q2'), get('l2')
        self.inv = np.argsort(self.perm, axis=1)
        self.b2 = (len(self.q2) - 1).bit_length()
        self.b1 = (len(self.q1) - 1).bit_length()
        self.bo = (len(self.pat) - 1).bit_length()
        self.bs = (len(self.perm) - 1).bit_length()
        self.ao = (get('aoi') / get('KS')).astype(np.float32)
        ae = get('aei') / get('KS')
        self.ae = np.concatenate([ae, -ae]).astype(np.float32)
        self.beta = np.float32(get('beta').item())
        ints = [self.inv, self.pat, self.lo, self.q1, self.l1, self.q2, self.l2]
        self.offsets = np.cumsum([0] + [v.size for v in ints[:-1]]).tolist()
        self.table = mx.array(np.concatenate([v.ravel() for v in ints]).astype(np.int32))
        # Divide each of the twelve possible magnitudes once, rather than per weight.
        self.levels = mx.array(np.concatenate([self.ao / self.beta, self.ae / self.beta]))
        aoi, aei, ks = get('aoi'), get('aei'), get('KS').item()
        self.integer = bool(np.all(aoi == np.round(aoi)) and np.all(aei == np.round(aei)))
        self.unit = float(np.float32(1 / (ks * float(self.beta))))
        self.forward = mx.array(self.perm.astype(np.uint32))
        if not self.integer:
            # CUDA mb10-style canonical decode: signs/patterns are folded once,
            # while input coordinates are permuted once per token, not per weight.
            sign7 = np.arange(128, dtype=np.uint32)
            signs = 1 - 2 * ((sign7[:, None] >> np.arange(7)) & 1).astype(np.int32)
            parity = np.array([int(v).bit_count() & 1 for v in sign7], np.int32)
            odd = np.empty((len(self.pat), 128, 8), np.float32)
            for o in range(len(self.pat)):
                s = np.concatenate([signs, (1 - 2 * ((parity + self.lo[o]) & 1))[:, None]], axis=1)
                odd[o] = (self.ao[self.pat[o]] / self.beta)[None, :] * s
            self.lut = mx.array(np.concatenate([odd.ravel(), (self.ae[self.q1] / self.beta).ravel(),
                                              (self.ae[self.q2] / self.beta).ravel()]))
            self.parities = mx.array(np.concatenate([self.l1, self.l2]).astype(np.uint32))
        else:
            odd_i = aoi[self.pat].astype(np.int8)
            odd_i[:, 7] *= (1 - 2 * (self.lo & 1)).astype(np.int8)
            even = np.concatenate([aei, -aei]).astype(np.int8)
            words = [odd_i.copy().view(np.uint32).reshape(-1, 2)]
            for q, parity in [(self.q1, self.l1), (self.q2, self.l2)]:
                words.append(np.stack([even[q].copy().view(np.uint32).ravel(), parity.astype(np.uint32)], -1))
            self.raw = mx.array(np.concatenate(words))

    @cached_property
    def defines(self):
        names = ['INV', 'PAT', 'LO', 'Q1', 'L1', 'Q2', 'L2']
        values = dict(zip(names, self.offsets))
        values.update(B1=self.b1, B2=self.b2, BO=self.bo, BS=self.bs)
        return '\n'.join(f'#define CQ_{k} {v}' for k, v in values.items()) + f'\n#define CQ_UNIT {self.unit:.9e}f\n'


def books(checkpoint):
    b = {p: Book(checkpoint, p) for p in ('ffq16', 'ffq15', 'ffq2')}
    return {'1-ffq': b['ffq16'], '15-ffq': b['ffq15'], '2-ffq': b['ffq2'], '4-ffq': b['ffq2']}
