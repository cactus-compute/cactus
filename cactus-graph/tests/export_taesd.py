"""Export TAESD decoder weights and reference artifacts for test_vae.cpp.

Usage: python export_taesd.py [output_dir] [test_image]
Then:  CACTUS_TAESD_DIR=<output_dir> ./test_vae

Requires torch, safetensors, and Pillow. Downloads the MIT-licensed TAESD
weights (https://huggingface.co/madebyollin/taesd) on first run.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "taesd"
ASSET = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(__file__).resolve().parents[2] / "cactus-engine" / "tests" / "assets" / "test_monkey.png"
OUT.mkdir(parents=True, exist_ok=True)

HF = "https://huggingface.co/madebyollin/taesd/resolve/main"


def fetch(name):
    dst = OUT / name
    if not dst.exists():
        print(f"downloading {name}...")
        urllib.request.urlretrieve(f"{HF}/{name}", dst)
    return dst


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )


def decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


enc = encoder()
enc.load_state_dict(load_file(fetch("taesd_encoder.safetensors")))
dec = decoder()
dec.load_state_dict(load_file(fetch("taesd_decoder.safetensors")))
enc.eval()
dec.eval()

img = Image.open(ASSET).convert("RGB").resize((512, 512), Image.LANCZOS)
x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

with torch.no_grad():
    latent = enc(x)
    ref = dec(latent).clamp(0, 1)

latent.to(torch.float16).numpy().tofile(OUT / "latent.f16.bin")
ref.to(torch.float32).numpy().tofile(OUT / "ref_image.f32.bin")
Image.fromarray((ref[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(OUT / "ref_image.png")

manifest = []
with open(OUT / "decoder.f16.bin", "wb") as f:
    for name, p in dec.named_parameters():
        f.write(p.detach().to(torch.float16).numpy().tobytes())
        manifest.append({"name": name, "shape": list(p.shape)})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

print(f"latent shape {list(latent.shape)}, range [{latent.min():.3f}, {latent.max():.3f}]")
print(f"wrote {len(manifest)} tensors to {OUT}")
