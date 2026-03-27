#!/usr/bin/env python3
"""Convert wespeaker-voxceleb-resnet34-LM weights to CACT format."""
import struct
import numpy as np
import torch
from pathlib import Path
from pyannote.audio import Model

REF_DIR = Path(__file__).parent
OUT_DIR = REF_DIR / "cactus_weights"
OUT_DIR.mkdir(exist_ok=True)

CACTUS_MAGIC = 0x54434143
HEADER_SIZE = 84
ALIGNMENT = 32

def save_weight(path, tensor, precision="fp16"):
    if precision == "fp16":
        data = tensor.astype(np.float16)
        prec_code = 1
    else:
        data = tensor.astype(np.float32)
        prec_code = 2

    shape = data.shape
    ndim = len(shape)
    data_bytes = data.tobytes()

    with open(path, "wb") as f:
        f.write(struct.pack('<I', CACTUS_MAGIC))
        f.write(struct.pack('<I', 0))  # flags
        f.write(struct.pack('<I', ALIGNMENT))
        f.write(struct.pack('<I', ndim))
        for i in range(4):
            f.write(struct.pack('<Q', shape[i] if i < ndim else 0))
        f.write(struct.pack('<I', prec_code))
        f.write(struct.pack('<Q', len(data_bytes)))
        f.write(struct.pack('<Q', 0))  # scales bytes
        f.write(struct.pack('<I', 0))  # group size
        f.write(struct.pack('<I', 0))  # num groups
        f.write(struct.pack('<Q', shape[-1] if ndim > 0 else 0))

        current_pos = HEADER_SIZE
        aligned_pos = ((current_pos + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        f.write(b'\x00' * (aligned_pos - current_pos))

        f.write(data_bytes)

    print(f"  {path.stem:50s} {str(list(shape)):20s} {precision}")

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
model.eval()
sd = model.state_dict()

print(f"Converting {len(sd)} tensors...")

for name, tensor in sorted(sd.items()):
    if "num_batches_tracked" in name:
        continue

    clean_name = name.replace(".", "_")
    arr = tensor.detach().cpu().numpy()

    if "shortcut.0.weight" in name:
        C_out, C_in, _, _ = arr.shape
        padded = np.zeros((C_out, C_in, 3, 3), dtype=np.float32)
        padded[:, :, 1, 1] = arr[:, :, 0, 0]
        save_weight(OUT_DIR / f"{clean_name}.weights", padded, "fp16")
    elif "running_mean" in name or "running_var" in name:
        save_weight(OUT_DIR / f"{clean_name}.weights", arr, "fp32")
    elif "bn" in name:
        save_weight(OUT_DIR / f"{clean_name}.weights", arr, "fp32")
    else:
        save_weight(OUT_DIR / f"{clean_name}.weights", arr, "fp16")

with open(OUT_DIR / "config.txt", 'w') as f:
    f.write("model_type=wespeaker\n")

print(f"\nWrote {len(list(OUT_DIR.glob('*.weights')))} weight files to {OUT_DIR}")
