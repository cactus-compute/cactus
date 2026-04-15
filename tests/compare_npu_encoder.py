"""Compare NPU encoder output vs HF reference for a 48-frame chunk."""
import numpy as np, torch, struct, os

ASSETS = os.path.join(os.path.dirname(__file__), 'assets')
WEIGHTS = r"C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\weights\gemma-4-e2b-it"

# Load the 100-frame mel and take first 48 frames
mel_all = np.load(os.path.join(ASSETS, 'audio_test_mel_input.npy'))  # [1, 100, 128]
mel48 = torch.tensor(mel_all[:, :48, :], dtype=torch.bfloat16)  # [1, 48, 128]
mel_mask = torch.zeros(1, 48, dtype=torch.bool)

print(f"mel48 shape: {mel48.shape}")

print("Loading HF model...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "gg-hf-gg/gemma-4-e2b-it", dtype=torch.bfloat16, device_map='cpu'
)
model.eval()
at = model.model.audio_tower

hooks = {}
def mkhook(name):
    def h(m, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hooks[name] = x.detach().float()
    return h

at.subsample_conv_projection.register_forward_hook(mkhook('sscp'))
for i, layer in enumerate(at.conformer):
    layer.register_forward_hook(mkhook(f'L{i}'))

with torch.no_grad():
    enc_out, _ = at(mel48, mel_mask)

print(f"HF encoder output shape: {enc_out.shape}")
print(f"HF sscp shape: {hooks['sscp'].shape}")
print(f"HF L0 shape: {hooks['L0'].shape}")

# Save reference for 48-frame chunk
ref48 = enc_out.squeeze(0).float().numpy()
ref48.tofile(os.path.join(ASSETS, 'audio_ref_enc48.bin'))
print(f"\nHF encoder (48 frames) first token first 8: {ref48[0, :8]}")
print(f"HF encoder (48 frames) norm: {np.linalg.norm(ref48[0]):.4f}")

sscp_ref = hooks['sscp'].squeeze(0).numpy()  # [12, 1024]
sscp_ref.tofile(os.path.join(ASSETS, 'audio_ref_sscp48.bin'))
print(f"\nHF SSCP (48 frames) first token first 8: {sscp_ref[0, :8]}")
print(f"HF SSCP (48 frames) norm: {np.linalg.norm(sscp_ref[0]):.4f}")

for i in range(12):
    lr = hooks[f'L{i}'].squeeze(0).numpy()
    lr.tofile(os.path.join(ASSETS, f'audio_ref_L{i:02d}_48.bin'))

print("\nSaved per-chunk reference files.")
print(f"\nPer-layer norms (tok0) for 48-frame HF reference:")
for i in range(12):
    h = hooks[f'L{i}'].squeeze(0)
    print(f"  L{i:02d}: ||tok0||={float(h[0].norm()):.4f}, first4={h[0,:4].numpy()}")
