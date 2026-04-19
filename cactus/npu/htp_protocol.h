#pragma once
// Wire protocol between cactus host and libggml-htp-vNN.so DSP skel.
// Matches htp-ops.h from llama.cpp ggml-hexagon backend exactly.
// The DSP skel is built from llama.cpp sources; this header must stay in sync.

#include <stdint.h>

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// Data types — must match GGML type enum (first values identical).
enum htp_data_type {
    HTP_TYPE_F32    =  0,
    HTP_TYPE_F16    =  1,
    HTP_TYPE_Q4_0   =  2,
    HTP_TYPE_Q8_0   =  8,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,
    HTP_TYPE_INVALID,
};

enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_RMS_NORM,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_ARGSORT,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_INVALID,
};

#define HTP_OP_MAX_DIMS    4
#define HTP_OP_MAX_INPUTS  6
#define HTP_OP_MAX_PARAMS  16
#define HTP_OP_MAX_BUFS    8
#define HTP_OP_MAX_REQS    256

// Tensor descriptor (sent in the dspqueue batch message).
struct htp_tensor {
    uint32_t data;                    // Byte offset within the buffer (bi)
    uint32_t size;                    // Data size in bytes
    uint32_t flags;                   // HTP_TENSOR_* flags
    uint16_t type;                    // htp_data_type
    uint16_t bi;                      // Buffer index (into htp_buf_desc array)
    uint32_t ne[HTP_OP_MAX_DIMS];     // Elements per dimension
    uint32_t nb[HTP_OP_MAX_DIMS];     // Stride in bytes per dimension
};

// GGML tensor flags relevant to us
#define HTP_TENSOR_COMPUTE  (1u << 0)  // Temporal compute data (not a weight)
#define HTP_TENSOR_FLUSHED  (1u << 1)  // Buffer has been flushed by the NPU

// Buffer descriptor (one per rpcmem region).
struct htp_buf_desc {
    uint64_t base;   // Host virtual address of the rpcmem region
    uint64_t size;   // Total size in bytes
    uint32_t flags;
    uint32_t fd;     // rpcmem file descriptor
};

// Op descriptor.
struct htp_op_desc {
    uint32_t opcode;                     // htp_op_code
    uint32_t flags;
    int32_t  params[HTP_OP_MAX_PARAMS];  // Op-specific params (e.g. epsilon for RMS norm)
    uint16_t src[HTP_OP_MAX_INPUTS];     // Source tensor indices
    uint16_t dst;                        // Destination tensor index

    // Filled by NPU:
    uint32_t prof_usecs;
    uint32_t prof_cycles;
    uint32_t prof_pkts;
    uint32_t unused;
};

// Batch request header — sent as the dspqueue message payload.
// Followed in the dspqueue buffer by:
//   htp_buf_desc  bufs    [n_bufs]
//   htp_tensor    tensors [n_tensors]
//   htp_op_desc   ops     [n_ops]
struct htp_opbatch_req {
    uint32_t n_bufs;
    uint32_t n_tensors;
    uint32_t n_ops;
    uint32_t flags;
};

// Batch response header — received as the dspqueue response payload.
struct htp_opbatch_rsp {
    uint32_t status;  // htp_status
};

// RoPE mode param (matches ggml ROPE_TYPE_*)
#define HTP_ROPE_TYPE_NEOX 2

// FLASH_ATTN_EXT params layout (int32 reinterpretation of float params):
//   params[0] = scale (float bits)
//   params[1] = max_bias (float bits)
//   params[2] = logit_softcap (float bits)
//   params[3] = attn_type (0=MHA, 1=MQA, ...)
//   params[4] = prec (0=default, 1=f32, ...)
