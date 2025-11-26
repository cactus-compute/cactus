import numpy as np

arr = np.load("encoder_output_manual.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("encoder_output_manual")
print("Shape:", arr.shape)
print(arr[:5,:5])


arr = np.load("encoder_block1_out.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("encoder_block1_out")
print("Shape:", arr.shape)
print(arr[:5,:5])

arr = np.load("encoder_block0_out.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("encoder_block0_out")
print("Shape:", arr.shape)
print(arr[:5,:5])

arr = np.load("encoder_block23_out.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("encoder_block23_out")
print("Shape:", arr.shape)
print(arr[:5,:5])


arr = np.load("decoder_emb_pos_ref.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("decoder_emb_pos_ref")
print("Shape:", arr.shape)
print(arr[:5,:5])


arr = np.load("decoder_post_norm_ref.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]
print("decoder_post_norm_ref")
print("Shape:", arr.shape)
print(arr[:5,:5])



# arr = np.load("encoder_block0_post_gelu.npy")

# # REMOVE batch dimension
# if arr.ndim == 3:
#     arr = arr[0]
# print("Encoder post gelu")
# print("Shape:", arr.shape)
# print(arr[:5,:5])



