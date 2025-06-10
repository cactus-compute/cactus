#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

LM_GGML_API void lm_ggml_critical_section_start(void);
LM_GGML_API void lm_ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif
