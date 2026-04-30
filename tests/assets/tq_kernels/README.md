# TQ Kernel Fixtures

These binary fixtures exercise the isolated TQ2/TQ4 GEMV and GEMM kernels in
the same shape as the model runtime path.

Each fixture contains:

1. A `TQFX` v4 header.
2. A fp16 codebook.
3. Raw per-K `input_scale`.
4. Hadamard `left_signs`, `right_signs`, and `permutation`.
5. Per-output-row/per-group norms.
6. Packed TQ indices in code-ordered layout.
7. Raw fp16 activations.
8. Expected fp16 outputs.

The public isolated kernels receive raw activations. They must fold
`1 / input_scale` into activations, apply the randomized Hadamard activation
rotation, read packed codebook indices, multiply row/group norms, and write the
final fp16 output. They must not consume pre-rotated activations or
pre-dequantized weights.

Regenerate fixtures with any Python that has NumPy available:

```sh
python3 tests/generate_tq_kernel_fixtures.py --out tests/assets/tq_kernels
```

Build and run only this test:

```sh
cmake -S tests -B tests/build
cmake --build tests/build --target test_tq_kernels -j
CACTUS_TEST_ASSETS=/Users/karen/cactus/tests/assets tests/build/test_tq_kernels
```
