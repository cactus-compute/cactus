"""Compute reference encoder output using cactus weights directly in PyTorch."""
import numpy as np, torch, torch.nn.functional as F, struct, math, os

WEIGHTS = r"C:\Users\justi\GitRepos\qualcomm-npu\third_party\cactus\weights\gemma-4-e2b-it"
ASSETS = os.path.join(os.path.dirname(__file__), 'assets')

def read_cactus(path):
    with open(path, 'rb') as f:
        data = f.read()
    _, _, align, ndim = struct.unpack_from('<4I', data, 0)
    shape = [struct.unpack_from('<Q', data, 16 + i*8)[0] for i in range(4)]
    shape = [d for i, d in enumerate(shape) if i < ndim and d > 0]
    prec, = struct.unpack_from('<I', data, 48)
    byte_size, = struct.unpack_from('<Q', data, 52)
    header_size = 84
    data_offset = math.ceil(header_size / align) * align if align > 0 else header_size
    raw = data[data_offset:data_offset + byte_size]
    if prec == 1:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif prec == 2:
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif prec == 4:
        arr16 = np.frombuffer(raw, dtype=np.uint16)
        arr32 = arr16.astype(np.uint32) << 16
        arr = arr32.view(np.float32).copy()
    else:
        # INT4 interleaved - read as uint8
        arr = np.frombuffer(raw, dtype=np.uint8)
        return arr.reshape(shape) if shape else arr, shape, prec
    return arr.reshape(shape) if shape else arr, shape, prec

def W(name):
    path = os.path.join(WEIGHTS, name + '.weights')
    t, shape, prec = read_cactus(path)
    if prec not in (1, 2, 4):
        # INT4 - need full dequant. For now flag it.
        print(f"  WARNING: {name} is INT4 ({prec}), skipping dequant")
        return None, shape
    return torch.tensor(t, dtype=torch.float32), shape

def rms_norm(x, w, eps=1e-6):
    # x: [T, D], w: [D]
    rms = (x ** 2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * w

def layer_norm(x, w, eps=0.001):
    return F.layer_norm(x, [x.shape[-1]], weight=w, eps=eps)

# Load mel input (100 frames), take first 48
mel_all = np.load(os.path.join(ASSETS, 'audio_test_mel_input.npy'))  # [1, 100, 128]
mel48 = torch.tensor(mel_all[0, :48, :], dtype=torch.float32)  # [48, 128]
print(f"mel48: {mel48.shape}, mean={mel48.mean():.4f}")

# ===== SSCP =====
mf = "audio_subsample_conv_projection_"

# Conv0: input [48, 128] as [1, 1, 48, 128] NCHW
w0, s0 = W(mf + "conv_0_conv")  # [128, 1, 3, 3]
n0, _ = W(mf + "conv_0_norm")[0], None  # [128]
n0 = W(mf + "conv_0_norm")[0]

x = mel48.unsqueeze(0).unsqueeze(0)  # [1, 1, 48, 128]
x0 = F.conv2d(x, w0, stride=2, padding=1)  # [1, 128, 24, 64]
H1, W1 = x0.shape[2], x0.shape[3]
# Reshape to [H1*W1, 128], layer norm, relu
x0f = x0[0].permute(1, 2, 0).reshape(H1*W1, -1)  # [H1*W1, 128]
x0f = layer_norm(x0f, n0)
x0f = F.relu(x0f)
x0r = x0f.reshape(H1, W1, -1)  # [H1, W1, 128]
print(f"After conv0+norm+relu: {x0r.shape}, H1={H1}, W1={W1}")

# Conv1: [1, 128, H1, W1] → [1, 32, H2, W2]
w1, s1 = W(mf + "conv_1_conv")  # [32, 128, 3, 3]
n1 = W(mf + "conv_1_norm")[0]   # [32]
x0_4d = x0r.permute(2, 0, 1).unsqueeze(0)  # [1, 128, H1, W1]
x1 = F.conv2d(x0_4d, w1, stride=2, padding=1)  # [1, 32, H2, W2]
H2, W2 = x1.shape[2], x1.shape[3]
x1f = x1[0].permute(1, 2, 0).reshape(H2*W2, -1)  # [H2*W2, 32]
x1f = layer_norm(x1f, n1)
x1f = F.relu(x1f)
x1r = x1f.reshape(H2, W2, -1)  # [H2, W2, 32]
print(f"After conv1+norm+relu: {x1r.shape}, H2={H2}, W2={W2}")

# Input projection: [H2, W2*32] → [H2, 1024]
ip, _ = W(mf + "input_proj")  # [out, in] or [in, out]?
print(f"input_proj shape: {ip.shape}")
xf = x1r.reshape(H2, W2 * x1r.shape[2])  # [12, 1024]
# op_matmul_T means matmul(x, w^T) = x @ w.T
h = xf @ ip.T
print(f"SSCP output: {h.shape}")
print(f"  tok0 first 8: {h[0, :8].numpy()}")
print(f"  tok0 norm: {h[0].norm():.4f}")

# Save SSCP reference
h.float().numpy().tofile(os.path.join(ASSETS, 'cactus_ref_sscp.bin'))
print(f"Saved cactus_ref_sscp.bin")

# ===== Conformer layers =====
cf_prefix = "audio_conformer_"
K_SCALE = math.log(1 + math.exp(1)) / math.log(2)
RMS_EPS = 1e-6
LOGIT_CAP = 50.0
RESIDUAL = 0.5
HIDDEN = 1024
NH = 8
HD = 128
CONF_K = 5

def conformer_ffw(h, li, is_end):
    tag = "end" if is_end else "start"
    wp = f"{cf_prefix}{li}_ffw_layer_{tag}_"
    pre_w = W(wp + "pre_layer_norm")[0]
    post_w = W(wp + "post_layer_norm")[0]
    w1_w = W(wp + "ffw_layer_1")[0]   # [4096, 1024]
    w2_w = W(wp + "ffw_layer_2")[0]   # [1024, 4096]
    if any(x is None for x in [pre_w, post_w, w1_w, w2_w]):
        return h
    x = rms_norm(h, pre_w, RMS_EPS)
    x = x @ w1_w.T            # [T, 4096]
    x = x * torch.sigmoid(x)  # SiLU
    x = x @ w2_w.T            # [T, 1024]
    x = rms_norm(x, post_w, RMS_EPS)
    return h + RESIDUAL * x

def conformer_attention(h, li):
    wp = f"{cf_prefix}{li}_attention_"
    pre_w = W(wp + "pre_attn_norm")[0]
    post_w = W(wp + "post_norm")[0]
    wq = W(wp + "attn_q_proj")[0]   # [1024, 1024]
    wk = W(wp + "attn_k_proj")[0]
    wv = W(wp + "attn_v_proj")[0]
    wo = W(wp + "post")[0]
    pds_w = W(wp + "attn_per_dim_scale")[0]  # [128]
    if any(x is None for x in [pre_w, post_w, wq, wk, wv, wo, pds_w]):
        return h

    T = h.shape[0]
    x = rms_norm(h, pre_w, RMS_EPS)
    q = x @ wq.T  # [T, 1024]
    k = x @ wk.T
    v = x @ wv.T

    # Per-dim scale
    q_scale = (1.0 / math.sqrt(HD)) / math.log(2.0)
    pds = torch.log(1 + torch.exp(pds_w)) * q_scale  # [128] softplus * q_scale
    # apply pds to q: q [T, 1024] = [T, NH, HD]; pds [HD]
    q3 = q.reshape(T, NH, HD) * pds.unsqueeze(0).unsqueeze(0)  # [T, NH, HD]
    q = q3.reshape(T, HIDDEN)

    # K scale
    k = k * K_SCALE

    # Reshape to [NH, T, HD]
    q_T = q.reshape(T, NH, HD).permute(1, 0, 2)  # [NH, T, HD]
    k_T = k.reshape(T, NH, HD).permute(1, 0, 2)
    v_T = v.reshape(T, NH, HD).permute(1, 0, 2)

    # QK^T scores [NH, T, T]
    scores = q_T @ k_T.transpose(-1, -2)  # [NH, T, T]

    # Logit cap
    scores = LOGIT_CAP * torch.tanh(scores / LOGIT_CAP)

    # Causal mask
    mask = torch.full((T, T), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask.unsqueeze(0)

    w = F.softmax(scores, dim=-1)  # [NH, T, T]
    out = (w @ v_T).permute(1, 0, 2).reshape(T, HIDDEN)  # [T, 1024]
    proj = out @ wo.T
    proj = rms_norm(proj, post_w, RMS_EPS)
    return h + proj

def conformer_lconv(h, li):
    wp = f"{cf_prefix}{li}_lconv1d_"
    pre_w = W(wp + "pre_layer_norm")[0]
    cnorm_w = W(wp + "conv_norm")[0]
    wstart = W(wp + "linear_start")[0]  # [2048, 1024]
    wend = W(wp + "linear_end")[0]       # [1024, 1024]
    dw_arr, dw_shape, dw_prec = read_cactus(os.path.join(WEIGHTS, wp + "depthwise_conv1d.weights"))
    if any(x is None for x in [pre_w, cnorm_w, wstart, wend]):
        return h
    print(f"  L{li} dw shape: {dw_shape}, prec: {dw_prec}")

    T = h.shape[0]
    x = rms_norm(h, pre_w, RMS_EPS)
    x = x @ wstart.T  # [T, 2048]
    # GLU: split into [T, 1024] x2, gate = first * silu(second)
    a, b = x[:, :HIDDEN], x[:, HIDDEN:]
    x = a * (b * torch.sigmoid(b))  # [T, 1024]

    if dw_prec in (1, 2, 4):
        dw = torch.tensor(dw_arr.astype(np.float32), dtype=torch.float32)
        # dw shape from file: [HIDDEN, 1, K] = [1024, 1, 5] → permute to [K, 1, 1, HIDDEN] for QNN
        # For PyTorch depthwise conv1d: weight shape [HIDDEN, 1, K]
        print(f"    dw torch shape: {dw.shape}")
        if len(dw.shape) == 3:
            # [HIDDEN, 1, K] → already correct for torch conv1d groups=HIDDEN
            pass
        # Causal padding: pad K-1 zeros on left
        K = dw.shape[-1] if len(dw.shape) == 3 else CONF_K
        x1d = x.unsqueeze(0).transpose(1, 2)  # [1, HIDDEN, T]
        x1d = F.pad(x1d, (K-1, 0))  # pad left
        x1d = F.conv1d(x1d, dw, groups=HIDDEN)  # [1, HIDDEN, T]
        x = x1d[0].transpose(0, 1)  # [T, HIDDEN]
    else:
        print(f"  L{li} depthwise conv skipped (INT4)")

    x = rms_norm(x, cnorm_w, RMS_EPS)
    x = x * torch.sigmoid(x)  # SiLU
    x = x @ wend.T
    return x + h

for li in range(12):
    block_norm_w = W(f"{cf_prefix}{li}_norm")[0]
    h = conformer_ffw(h, li, False)
    h = conformer_attention(h, li)
    h = conformer_lconv(h, li)
    h = conformer_ffw(h, li, True)
    if block_norm_w is not None:
        h = rms_norm(h, block_norm_w, RMS_EPS)
    print(f"L{li:02d}: tok0 norm={h[0].norm():.4f}, first4={h[0,:4].numpy()}")
    h.float().numpy().tofile(os.path.join(ASSETS, f'cactus_ref_L{li:02d}.bin'))

# Output projection
op_w, _ = W("audio_output_proj")
# bias file has different extension
def read_bias(name):
    path = os.path.join(WEIGHTS, name + '.bias')
    if not os.path.exists(path):
        return None
    t, shape, prec = read_cactus(path)
    if prec not in (1, 2, 4): return None
    return torch.tensor(t.astype(np.float32), dtype=torch.float32)

ob_w = read_bias("audio_output_proj")
if op_w is not None:
    h_out = h @ op_w.T
    if ob_w is not None:
        h_out = h_out + ob_w
    print(f"\nOutput proj: {h_out.shape}, tok0 first 8: {h_out[0,:8].numpy()}")
    h_out.float().numpy().tofile(os.path.join(ASSETS, 'cactus_ref_encoder.bin'))
else:
    print("No output_proj found, saving conformer output")
    h.float().numpy().tofile(os.path.join(ASSETS, 'cactus_ref_encoder.bin'))

print("\nDone. Saved cactus_ref_*.bin files")
