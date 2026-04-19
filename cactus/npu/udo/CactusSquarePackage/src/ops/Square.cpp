#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_Square);

template <typename T_Ttype>
int squareImplFp16(T_Ttype &out, const T_Ttype &in);

DEF_TENSOR_PROPERTIES(Op("Square", "in0"), Crouton("*", "in0"), MainMemory("*", "in0"))

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((squareImplFp16<F16CroutonTensor>),
                                  "Square",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((squareImplFp16<PlainFloat16Tensor>),
                                  "Square",
                                  FAST,
                                  Flags::RESOURCE_HVX)

template <typename T_Ttype>
int squareImplFp16(T_Ttype &out, const T_Ttype &in) {
  // debuglog(...)  // disabled for MSVC host build
  out.set_dims(in);

  size_t inBlocks  = in.blocktab_len();
  auto   inBlocktab  = in.blocktab_ptr();
  auto   outBlocktab = out.blocktab_ptr();

  for (uint32_t i = 0; i < inBlocks; ++i) {
    auto inVptr  = (const HVX_Vector *)(inBlocktab[i]);
    auto outVptr = (HVX_Vector *)(outBlocktab[i]);
    for (uint32_t j = 0; j < 16; ++j) {
      HVX_Vector vin = inVptr[j];
      HVX_Vector vqf = Q6_Vqf16_vmpy_VhfVhf(vin, vin);
      outVptr[j]     = Q6_Vhf_equals_Vqf16(vqf);
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_Square);
