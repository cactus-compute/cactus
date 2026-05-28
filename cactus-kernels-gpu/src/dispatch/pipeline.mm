/* Cactus GPU pipeline (compute kernel pipeline state) — Metal compile +
 * function-constant binding. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include "internal.h"

#include <cstdio>
#include <cstring>

namespace cactus {
namespace gpu {

#if !CACTUS_HAS_METAL

Pipeline* pipeline_create(Context*, const char*, const FunctionConstant*, size_t) { return nullptr; }
void pipeline_destroy(Pipeline*) {}

#else

static MTLDataType fc_type_to_metal(FCType t) {
    switch (t) {
        case FCType::BOOL:   return MTLDataTypeBool;
        case FCType::INT32:  return MTLDataTypeInt;
        case FCType::UINT32: return MTLDataTypeUInt;
        case FCType::FP32:   return MTLDataTypeFloat;
    }
    return MTLDataTypeFloat;
}

Pipeline* pipeline_create(Context* ctx,
                          const char* kernel_name,
                          const FunctionConstant* constants,
                          size_t num_constants) {
    if (!ctx || !kernel_name) return nullptr;
    @autoreleasepool {
        id<MTLDevice>  dev = _internal_device(ctx);
        id<MTLLibrary> lib = _internal_library(ctx);

        MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
        for (size_t i = 0; i < num_constants; ++i) {
            const FunctionConstant& c = constants[i];
            switch (c.type) {
                case FCType::BOOL:
                    [fcv setConstantValue:&c.value.b type:MTLDataTypeBool atIndex:c.index];
                    break;
                case FCType::INT32:
                    [fcv setConstantValue:&c.value.i32 type:MTLDataTypeInt atIndex:c.index];
                    break;
                case FCType::UINT32:
                    [fcv setConstantValue:&c.value.u32 type:MTLDataTypeUInt atIndex:c.index];
                    break;
                case FCType::FP32:
                    [fcv setConstantValue:&c.value.f32 type:MTLDataTypeFloat atIndex:c.index];
                    break;
            }
        }

        NSError* error = nil;
        NSString* name = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> fn = [lib newFunctionWithName:name
                                       constantValues:fcv
                                                error:&error];
        if (!fn) {
            std::fprintf(stderr, "cactus_gpu: newFunctionWithName(%s) failed: %s\n",
                         kernel_name,
                         error ? [[error localizedDescription] UTF8String] : "?");
            return nullptr;
        }

        id<MTLComputePipelineState> ps =
            [dev newComputePipelineStateWithFunction:fn error:&error];
        if (!ps) {
            std::fprintf(stderr, "cactus_gpu: pipeline(%s) failed: %s\n",
                         kernel_name,
                         error ? [[error localizedDescription] UTF8String] : "?");
            return nullptr;
        }

        Pipeline* out = new Pipeline();
        out->pipeline_state    = ps;
        out->function          = fn;
        out->tg_threads        = (uint32_t)[ps maxTotalThreadsPerThreadgroup];
        out->simdgroup_threads = (uint32_t)[ps threadExecutionWidth];
        return out;
    }
}

void pipeline_destroy(Pipeline* p) {
    if (!p) return;
    @autoreleasepool { p->pipeline_state = nil; p->function = nil; }
    delete p;
}

#endif

} // namespace gpu
} // namespace cactus
