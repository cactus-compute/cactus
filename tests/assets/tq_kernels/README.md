# TQ Kernel Inputs

`test_tq_kernels` loads real converted Gemma4 TQ weights directly from:

```sh
weights/gemma-4-e2b-it-tqh-u4-codeorder
```

Override that path with:

```sh
CACTUS_TQ_WEIGHTS_ROOT=/path/to/weights
```

The files in this directory are only raw fp16 activation inputs. They do not
contain packed weights or expected outputs. Expected outputs are computed in the
test by a scalar reference path from the same loaded weight slice.

Activation file format:

```text
magic:   "TQAC"
version: uint32 = 1
M:       uint32
K:       uint32
data:    fp16[M * K]
```

Regenerate activation inputs:

```sh
python3 tests/generate_tq_activation_assets.py \
  --weights-root weights/gemma-4-e2b-it-tqh-u4-codeorder \
  --out tests/assets/tq_kernels
```

Run the test:

```sh
cmake -S tests -B tests/build
cmake --build tests/build --target test_tq_kernels -j
CACTUS_TEST_ASSETS=/Users/karen/cactus/tests/assets tests/build/test_tq_kernels
```
