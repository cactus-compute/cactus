# Python Bindings

ctypes FFI bindings to `cactus_engine.h`. The package source lives at [`/python/`](/python/).

## Integration

```bash
cactus build --python
pip install -e python/
```

## Usage

```python
from cactus.bindings import cactus

model = cactus.cactus_init("/path/to/model", None, False)
response = cactus.cactus_complete(model, messages_json, None, None, None)
cactus.cactus_destroy(model)
```
