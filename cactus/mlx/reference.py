"""Independent NumPy decoder used for parity checks, never full-model loading."""
import numpy as np
from .checkpoint import RATE_BYTES


def hadamard(x):
    x = np.array(x, dtype=np.float32, copy=True)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        shape = x.shape
        pairs = x.reshape(*shape[:-1], -1, 2, stride)
        a, b = pairs[..., 0, :].copy(), pairs[..., 1, :].copy()
        pairs[..., 0, :], pairs[..., 1, :] = a + b, a - b
    return x


def decode_codes(codes, rate, book):
    packed = np.asarray(codes).reshape(-1, RATE_BYTES[rate])
    if rate == '4-ffq':
        u16 = packed.copy().view('<u2').reshape(-1, 8, 4).astype(np.uint32)
        stages = [u16[..., s] | (u16[..., s + 2] << 16) for s in (0, 1)]
    else:
        width = RATE_BYTES[rate] // 8
        raw = packed.reshape(-1, 8, width).astype(np.uint32)
        stages = [np.sum(raw << (8 * np.arange(width, dtype=np.uint32)), axis=-1, dtype=np.uint32)]
    decoded = []
    for cw in stages:
        i2 = cw & ((1 << book.b2) - 1)
        i1 = (cw >> book.b2) & ((1 << book.b1) - 1)
        s7 = (cw >> (book.b2 + book.b1)) & 127
        o = (cw >> (book.b2 + book.b1 + 7)) & ((1 << book.bo) - 1)
        slot = cw >> (book.b2 + book.b1 + 7 + book.bo)
        signs = ((s7[..., None] >> np.arange(7)) & 1).astype(np.int32)
        last = (book.lo[o] + signs.sum(-1) + (book.l1[i1] ^ book.l2[i2])) & 1
        sign = 1 - 2 * np.concatenate([signs, last[..., None]], -1)
        odd = book.ao[book.pat[o]] * sign.astype(np.float32)
        canonical = np.concatenate([odd, book.ae[book.q1[i1]], book.ae[book.q2[i2]]], -1)
        dec = np.take_along_axis(canonical, book.inv[slot], axis=-1) / book.beta
        decoded.append(dec.reshape(-1, 128))
    return decoded[0] if len(decoded) == 1 else decoded[0] / np.float32(.93) + np.float32(.28) * decoded[1]


def decode_rows(checkpoint, name, ids):
    from .checkpoint import books
    leaf = checkpoint.leaves[name]
    data = checkpoint.rows(name, ids)
    dec = decode_codes(data['codes'], leaf['rate'], books(checkpoint)[leaf['rate']])
    dec = dec.reshape(len(ids), -1, 128)
    dec *= (data['norms'] * data['norm_scale'][:, None] / np.float32(np.sqrt(128)))[..., None]
    weights = hadamard(dec) * np.float32(1 / np.sqrt(128))
    return weights.reshape(len(ids), -1) * data['a'] * data['c'][:, None]
