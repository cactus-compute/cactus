# Python Bindings

ctypes FFI to `cactus_engine.h`. The binding module lives at
[`/python/cactus/bindings/cactus.py`](/python/cactus/bindings/cactus.py).

It loads `libcactus_engine.{so,dylib}` from `/python/cactus/bin/`, populated by
`cactus build --python`.

For installation, the high-level API, the CLI, the server, and the transpiler,
see [`/python/README.md`](/python/).
