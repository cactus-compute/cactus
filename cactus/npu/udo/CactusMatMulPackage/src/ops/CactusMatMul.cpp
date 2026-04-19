#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_CactusMatMul);

template <typename T_Ttype>
int cactusMatMulImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b);

DEF_TENSOR_PROPERTIES(Op("CactusMatMul", "a", "b"),
                      MainMemory("*", "a", "b"))

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((cactusMatMulImpl<PlainFloat16Tensor>),
                                  "CactusMatMul",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)

// 2D matmul over a 4D tensor shape [1, 1, M, K] x [1, 1, K, N] -> [1, 1, M, N].
// FP16 everywhere. Output elements computed with qf16 vector accumulator; N
// dim vectorised across the HVX multiply.
template <typename T_Ttype>
int cactusMatMulImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b) {
  const size_t M = a.dim(2);
  const size_t K = a.dim(3);
  const size_t K2 = b.dim(2);
  const size_t N = b.dim(3);
  debuglog("CactusMatMul: a=[1,1,%zu,%zu] b=[1,1,%zu,%zu] -> out=[1,1,%zu,%zu]",
           M, K, K2, N, M, N);

  if (K != K2) {
    errlog("CactusMatMul: inner dims mismatch K=%zu K2=%zu", K, K2);
    return GraphStatus::ErrorFatal;
  }
  const size_t out_dims[4] = {a.dim(0), a.dim(1), M, N};
  out.set_dims(out_dims);

  const uint16_t *A = reinterpret_cast<const uint16_t *>(a.raw_data_const());
  const uint16_t *B = reinterpret_cast<const uint16_t *>(b.raw_data_const());
  uint16_t       *C = reinterpret_cast<uint16_t *>(out.raw_data());

  // Vectorised across N dim (64 fp16 elements per HVX vector); scalar tail for
  // leftover columns.
  constexpr size_t VLEN_HF = 64;
  const size_t     N_vec   = (N / VLEN_HF) * VLEN_HF;

  for (size_t m = 0; m < M; ++m) {
    for (size_t n0 = 0; n0 < N_vec; n0 += VLEN_HF) {
      HVX_Vector acc;
      for (size_t k = 0; k < K; ++k) {
        HVX_Vector a_splat = Q6_Vh_vsplat_R(A[m * K + k]);
        HVX_Vector b_row   =
            *reinterpret_cast<const HVX_Vector *>(&B[k * N + n0]);
        HVX_Vector prod_qf = Q6_Vqf16_vmpy_VhfVhf(a_splat, b_row);
        acc = (k == 0) ? prod_qf : Q6_Vqf16_vadd_Vqf16Vqf16(acc, prod_qf);
      }
      *reinterpret_cast<HVX_Vector *>(&C[m * N + n0]) =
          Q6_Vhf_equals_Vqf16(acc);
    }

    for (size_t n = N_vec; n < N; ++n) {
      float accf = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        accf += float(Float16::from_raw(A[m * K + k]))
              * float(Float16::from_raw(B[k * N + n]));
      }
      C[m * N + n] = Float16(accf).raw();
    }
  }

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_CactusMatMul);
