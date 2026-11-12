import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from ..checkpoint import Checkpoint, books
from ..ops import CQLinear
from ..reference import decode_codes, decode_rows, hadamard


def error(actual, expected):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    return {'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            'max_abs': float(np.max(np.abs(a - b)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output')
    parser.add_argument('--matmul-tolerance', type=float, default=2e-5)
    args = parser.parse_args()
    checkpoint = Checkpoint(args.model)
    bs = books(checkpoint)
    cases = [
        ('model.layers.3.mixer.shared_experts.down_proj.weight', None),
        ('model.layers.1.mixer.experts.up.weight', [0, 17, 511]),
        ('model.layers.0.mixer.in_proj.weight', None),
        ('model.layers.7.mixer.v_proj.weight', None),
        ('model.layers.1.mixer.experts.down.weight', [0, 17, 511]),
    ]
    results = []
    rng = np.random.default_rng(42)
    for name, experts in cases:
        start = time.perf_counter()
        leaf = checkpoint.leaves[name]
        op = CQLinear(checkpoint, name, bs[leaf['rate']])
        n = leaf['scale_rows'] if experts else op._n
        sample = np.array([0, 1, n // 2, n - 1])
        ids = sample if not experts else np.concatenate([sample + e * n for e in experts])
        reference = decode_rows(checkpoint, name, ids)
        precision = 'fp16'
        data = checkpoint.rows(name, ids)
        rotated_weights = decode_codes(data['codes'], leaf['rate'], bs[leaf['rate']]).reshape(len(ids),-1,128)
        rotated_weights *= (data['norms'] * data['norm_scale'][:,None] * data['c'][:,None])[...,None]
        rotated_weights = rotated_weights.reshape(len(ids),-1)

        def prepare_reference(x):
            rotated = hadamard((x * data['a']).reshape(*x.shape[:-1],-1,128)) / np.float32(128)
            rotated = rotated.astype(np.float16).astype(np.float32)
            return rotated.reshape(x.shape)
        gathered = op.gather(mx.array(ids), dtype=mx.float32)
        mx.eval(gathered)
        gather_error = error(np.array(gathered), reference)
        assert gather_error['relative_l2'] < 3e-6, (name, gather_error)
        checks = []
        for t in (1, 3):
            x = rng.normal(0, .25, (t, op._k)).astype(np.float32)
            x_mx = mx.array(x)
            if experts:
                indices = mx.array(np.tile(experts, (t, 1)), dtype=mx.uint32)
                output = op.matmul(x_mx, indices, broadcast=True, expert_rows=n)
                picked = output[..., mx.array(sample)].reshape(t, -1)
            else:
                output = op(x_mx)
                picked = output[..., mx.array(sample)]
            mx.eval(picked)
            expected = x @ reference.T
            e = error(np.array(picked), expected)
            arithmetic = error(np.array(picked), prepare_reference(x) @ rotated_weights.T)
            assert arithmetic['relative_l2'] < args.matmul_tolerance, (name, t, arithmetic)
            e['arithmetic_reference'] = arithmetic
            checks.append({'tokens': t, **e})
        if experts:
            # Down projection receives one distinct activation for every selected expert.
            x = rng.normal(0, .25, (2, len(experts), op._k)).astype(np.float32)
            indices = mx.array(np.tile(experts, (2, 1)), dtype=mx.uint32)
            output = op.matmul(mx.array(x), indices, expert_rows=n)[..., mx.array(sample)]
            expected = np.einsum('tek,erk->ter', x, reference.reshape(len(experts), len(sample), -1))
            mx.eval(output)
            e = error(np.array(output), expected)
            expected_quantized = np.einsum('tek,erk->ter', prepare_reference(x), rotated_weights.reshape(len(experts),len(sample),-1))
            arithmetic = error(np.array(output), expected_quantized)
            assert arithmetic['relative_l2'] < args.matmul_tolerance, (name, arithmetic)
            e['arithmetic_reference'] = arithmetic
            checks.append({'distinct_expert_inputs': True, **e})
        result = {'name': name, 'rate': leaf['rate'], 'activation': precision, 'gather': gather_error,
                  'matmul': checks, 'elapsed_s': time.perf_counter() - start}
        results.append(result)
        print(json.dumps(result), flush=True)
        del op, output, picked, gathered
        gc.collect()
        mx.clear_cache()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({'device': mx.device_info(), 'checks': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
