# Python Bindings

ctypes FFI to `cactus_engine.h`. The binding module lives at
`/python/cactus/bindings/cactus.py`.

It loads `libcactus_engine.{so,dylib}` from `/python/cactus/bindings/lib/` (bundled)
or `cactus-engine/build/` (dev build), populated by `cactus build --python`.

For installation, the high-level API, the CLI, and the server,
see [`/python/README.md`](/python/).
