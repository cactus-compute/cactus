#include "npu_ane.h"

#if CACTUS_HAS_ANE

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include "../graph/graph.h"

namespace {

static std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool parse_compute_units(const char* raw_value, MLComputeUnits& out_units) {
    if (!raw_value || raw_value[0] == '\0') {
        return false;
    }

    std::string value = to_lower_ascii(raw_value);
    if (value == "all" || value == "default") {
        out_units = MLComputeUnitsAll;
        return true;
    }
    if (value == "cpu_and_ne" || value == "cpu-ne" || value == "cpu_ne" ||
        value == "ne" || value == "ane" || value == "cpuandneuralengine") {
        out_units = MLComputeUnitsCPUAndNeuralEngine;
        return true;
    }
    if (value == "cpu_and_gpu" || value == "cpu_gpu" || value == "cpu-gpu" ||
        value == "cpuandgpu") {
        out_units = MLComputeUnitsCPUAndGPU;
        return true;
    }
    if (value == "cpu_only" || value == "cpu-only" || value == "cpu") {
        out_units = MLComputeUnitsCPUOnly;
        return true;
    }

    return false;
}

static const char* compute_units_to_string(MLComputeUnits units) {
    switch (units) {
        case MLComputeUnitsAll:
            return "ALL";
        case MLComputeUnitsCPUAndGPU:
            return "CPU_AND_GPU";
        case MLComputeUnitsCPUOnly:
            return "CPU_ONLY";
        case MLComputeUnitsCPUAndNeuralEngine:
            return "CPU_AND_NE";
    }
    return "UNKNOWN";
}

static bool model_path_looks_like_audio_encoder(NSString* model_path) {
    if (!model_path) return false;
    NSString* lower = [[model_path lastPathComponent] lowercaseString];
    return [lower containsString:@"audio_encoder"];
}

static bool model_path_looks_like_vision_encoder(NSString* model_path) {
    if (!model_path) return false;
    NSString* lower = [[model_path lastPathComponent] lowercaseString];
    return [lower containsString:@"vision_encoder"];
}

static void maybe_apply_compute_units_env(const char* env_name, MLComputeUnits& target) {
    const char* raw = std::getenv(env_name);
    if (!raw || raw[0] == '\0') return;

    MLComputeUnits parsed;
    if (parse_compute_units(raw, parsed)) {
        target = parsed;
        return;
    }

    CACTUS_LOG_WARN("npu", "Ignoring invalid " << env_name << "=" << raw);
}

static MLComputeUnits resolve_compute_units_for_model(NSString* model_path, bool is_prefill) {
    MLComputeUnits units = MLComputeUnitsCPUAndNeuralEngine;

    // Gemma4 multimodal encoders show better fidelity on CPU+GPU.
    if (!is_prefill && model_path_looks_like_audio_encoder(model_path)) {
        units = MLComputeUnitsCPUAndGPU;
    }
    if (!is_prefill && model_path_looks_like_vision_encoder(model_path)) {
        units = MLComputeUnitsCPUAndGPU;
    }

    maybe_apply_compute_units_env("CACTUS_ANE_COMPUTE_UNITS", units);
    if (is_prefill) {
        maybe_apply_compute_units_env("CACTUS_ANE_PREFILL_COMPUTE_UNITS", units);
    } else {
        maybe_apply_compute_units_env("CACTUS_ANE_ENCODER_COMPUTE_UNITS", units);
        if (model_path_looks_like_audio_encoder(model_path)) {
            maybe_apply_compute_units_env("CACTUS_ANE_AUDIO_COMPUTE_UNITS", units);
        }
        if (model_path_looks_like_vision_encoder(model_path)) {
            maybe_apply_compute_units_env("CACTUS_ANE_VISION_COMPUTE_UNITS", units);
        }
    }

    return units;
}

static bool should_recompile_mlpackage(NSString* mlpackage_path, NSString* mlmodelc_path) {
    NSFileManager* fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:mlmodelc_path]) return true;

    const char* force = std::getenv("CACTUS_ANE_FORCE_RECOMPILE");
    if (force && force[0] != '\0' && std::strcmp(force, "0") != 0) {
        return true;
    }

    NSError* pkg_err = nil;
    NSError* cache_err = nil;
    NSDictionary* pkg_attr = [fm attributesOfItemAtPath:mlpackage_path error:&pkg_err];
    NSDictionary* cache_attr = [fm attributesOfItemAtPath:mlmodelc_path error:&cache_err];
    if (!pkg_attr || !cache_attr || pkg_err || cache_err) {
        return false;
    }

    NSDate* pkg_mtime = pkg_attr[NSFileModificationDate];
    NSDate* cache_mtime = cache_attr[NSFileModificationDate];
    if (!pkg_mtime || !cache_mtime) {
        return false;
    }
    return ([pkg_mtime compare:cache_mtime] == NSOrderedDescending);
}

static NSURL* resolve_or_compile_model_url(NSString* path, NSError** error) {
    NSURL* modelURL = [NSURL fileURLWithPath:path];
    if (![path hasSuffix:@".mlpackage"]) {
        return modelURL;
    }

    BOOL isDir = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir] || !isDir) {
        CACTUS_LOG_ERROR("npu", "ANE mlpackage path is not a valid directory: " << [path UTF8String]);
        return modelURL;
    }

    NSString* cachedPath = [[path stringByDeletingPathExtension] stringByAppendingPathExtension:@"mlmodelc"];
    NSURL* cachedURL = [NSURL fileURLWithPath:cachedPath];
    NSFileManager* fm = [NSFileManager defaultManager];

    if (!should_recompile_mlpackage(path, cachedPath)) {
        return cachedURL;
    }

    if ([fm fileExistsAtPath:cachedPath]) {
        NSError* rmErr = nil;
        [fm removeItemAtPath:cachedPath error:&rmErr];
        if (rmErr) {
            CACTUS_LOG_WARN("npu", "Failed to remove stale mlmodelc: " << [[rmErr localizedDescription] UTF8String]);
        } else {
            CACTUS_LOG_INFO("npu", "Removed stale mlmodelc cache, recompiling: " << [cachedPath UTF8String]);
        }
    }

    NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:error];
    if (*error || !compiledURL) {
        CACTUS_LOG_ERROR("npu", "ANE model compilation failed: " << [[*error localizedDescription] UTF8String]);
        return modelURL;
    }

    NSError* moveError = nil;
    [fm moveItemAtURL:compiledURL toURL:cachedURL error:&moveError];
    if (moveError) {
        CACTUS_LOG_WARN("npu", "Could not persist compiled mlmodelc cache: " << [[moveError localizedDescription] UTF8String]);
        return compiledURL;
    }
    return cachedURL;
}

} // namespace

@interface CactusANEImpl : NSObject

@property (nonatomic, strong) MLModel* model;
@property (nonatomic, strong) MLModelDescription* modelDescription;
@property (nonatomic, strong) MLMultiArray* cachedInputArray;
@property (nonatomic, strong) MLMultiArray* cachedOutputArray;
@property (nonatomic, strong) NSArray<NSNumber*>* cachedShape;
@property (nonatomic, strong) NSArray<NSNumber*>* cachedOutputShape;
@property (nonatomic, strong) NSString* cachedInputName;
@property (nonatomic, strong) NSString* cachedOutputName;
@property (nonatomic, assign) NSUInteger cachedOutputSize;
@property (nonatomic, strong) MLPredictionOptions* predictionOptions;
@property (nonatomic, strong) NSMutableDictionary<NSString*, MLMultiArray*>* cachedMultiInputArrays;
@property (nonatomic, strong) MLMultiArray* cachedMultiOutputArray;
@property (nonatomic, strong) MLPredictionOptions* multiPredictionOptions;
@property (nonatomic, assign) BOOL multiInputPreallocated;
@property (nonatomic, assign) BOOL hasOutputBackings;
@property (nonatomic, assign) BOOL hasMultiOutputBackings;

- (instancetype)initWithModelPath:(NSString*)path;
- (NSArray<NSNumber*>*)getInputShape;
- (NSArray<NSNumber*>*)getOutputShape;
- (BOOL)preallocateBuffersWithInput:(NSString*)inputName
                              shape:(NSArray<NSNumber*>*)shape
                         outputName:(NSString*)outputName;
- (BOOL)canUseCachedBufferWithInput:(NSString*)inputName
                              shape:(NSArray<NSNumber*>*)shape
                         outputName:(NSString*)outputName;
- (MLMultiArray*)predictWithInput:(NSString*)inputName
                             data:(const __fp16*)data
                            shape:(NSArray<NSNumber*>*)shape
                       outputName:(NSString*)outputName;
- (BOOL)preallocateMultiInputBuffersWithOutputName:(NSString*)outputName;
- (size_t)predictMultiInputWithDict:(NSMutableDictionary<NSString*, MLFeatureValue*>*)inputDict
                         outputData:(__fp16*)output
                         outputName:(NSString*)outputName;

@end

@implementation CactusANEImpl

- (instancetype)initWithModelPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSError* error = nil;
        NSURL* modelURL = resolve_or_compile_model_url(path, &error);

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = resolve_compute_units_for_model(path, false);
        CACTUS_LOG_INFO("npu",
                        "ANEEncoder compute units: "
                            << compute_units_to_string(config.computeUnits)
                            << " for " << [path UTF8String]);

        _model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (_model) {
            _modelDescription = _model.modelDescription;
                    }
        if (error) {
            CACTUS_LOG_ERROR("npu", "ANE model load failed: " << [[error localizedDescription] UTF8String]);
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)getInputShape {
    if (!_modelDescription) return @[];

    NSString* inputName = _cachedInputName;
    if (!inputName) {
        inputName = _modelDescription.inputDescriptionsByName.allKeys.firstObject;
    }

    MLFeatureDescription* inputDesc = _modelDescription.inputDescriptionsByName[inputName];
    if (inputDesc && inputDesc.type == MLFeatureTypeMultiArray) {
        return inputDesc.multiArrayConstraint.shape;
    }

    return @[];
}

- (NSArray<NSNumber*>*)getOutputShape {
    if (!_modelDescription) return @[];

    NSString* outputName = _cachedOutputName;
    if (!outputName) {
        outputName = _modelDescription.outputDescriptionsByName.allKeys.firstObject;
    }

    MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[outputName];
    if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
        return outputDesc.multiArrayConstraint.shape;
    }

    return @[];
}

- (BOOL)preallocateBuffersWithInput:(NSString*)inputName
                              shape:(NSArray<NSNumber*>*)shape
                         outputName:(NSString*)outputName {
    if (!_model) return NO;

    NSError* error = nil;

    MLMultiArrayDataType inputDataType = MLMultiArrayDataTypeFloat16;
    MLFeatureDescription* inputDesc = _modelDescription.inputDescriptionsByName[inputName];
    if (inputDesc && inputDesc.multiArrayConstraint) {
        inputDataType = inputDesc.multiArrayConstraint.dataType;
    }

    _cachedInputArray = [[MLMultiArray alloc]
        initWithShape:shape
             dataType:inputDataType
                error:&error];

    if (error) {
        CACTUS_LOG_ERROR("npu", "ANE preallocate input array failed: " << [[error localizedDescription] UTF8String]);
        return NO;
    }

    _cachedShape = [shape copy];
    _cachedInputName = [inputName copy];
    _cachedOutputName = outputName ? [outputName copy]
                                   : _modelDescription.outputDescriptionsByName.allKeys.firstObject;

    CACTUS_LOG_DEBUG("npu",
                     "[CactusANE] prealloc input name="
                         << [_cachedInputName UTF8String]
                         << " shape=" << [[_cachedInputArray.shape description] UTF8String]
                         << " strides=" << [[_cachedInputArray.strides description] UTF8String]
                         << " dtype=" << (long)_cachedInputArray.dataType);

    MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[_cachedOutputName];
    if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* outputShape = outputDesc.multiArrayConstraint.shape;
        MLMultiArrayDataType outputDataType = outputDesc.multiArrayConstraint.dataType;

        _cachedOutputArray = [[MLMultiArray alloc]
            initWithShape:outputShape
                 dataType:outputDataType
                    error:&error];

        if (error) {
            CACTUS_LOG_ERROR("npu", "ANE preallocate output array failed: " << [[error localizedDescription] UTF8String]);
            return NO;
        }

        _cachedOutputShape = [outputShape copy];
        _cachedOutputSize = 1;
        for (NSNumber* dim in outputShape) {
            _cachedOutputSize *= [dim unsignedIntegerValue];
        }

        CACTUS_LOG_DEBUG("npu",
                         "[CactusANE] prealloc output name="
                             << [_cachedOutputName UTF8String]
                             << " shape=" << [[_cachedOutputArray.shape description] UTF8String]
                             << " strides=" << [[_cachedOutputArray.strides description] UTF8String]
                             << " dtype=" << (long)_cachedOutputArray.dataType);

        _predictionOptions = [[MLPredictionOptions alloc] init];
        _hasOutputBackings = NO;
        if (_predictionOptions && _cachedOutputArray && _cachedOutputName) {
            _predictionOptions.outputBackings =
                @{ _cachedOutputName: _cachedOutputArray };
            _hasOutputBackings = YES;
        }
    }

    return YES;
}

- (BOOL)canUseCachedBufferWithInput:(NSString*)inputName
                              shape:(NSArray<NSNumber*>*)shape
                         outputName:(NSString*)outputName {
    if (!_cachedInputArray || !_cachedShape) return NO;
    if (![_cachedInputName isEqualToString:inputName]) return NO;
    if (_cachedShape.count != shape.count) return NO;

    for (NSUInteger i = 0; i < shape.count; i++) {
        if (![_cachedShape[i] isEqualToNumber:shape[i]]) return NO;
    }

    if (outputName && outputName.length > 0 && ![_cachedOutputName isEqualToString:outputName]) {
        return NO;
    }

    return YES;
}

- (MLMultiArray*)predictWithInput:(NSString*)inputName
                             data:(const __fp16*)data
                            shape:(NSArray<NSNumber*>*)shape
                       outputName:(NSString*)outputName {
    if (!_model) return nil;

    NSError* error = nil;
    MLMultiArray* inputArray = nil;

    MLMultiArrayDataType expectedDataType = MLMultiArrayDataTypeFloat16;
    MLFeatureDescription* inputDesc = _modelDescription.inputDescriptionsByName[inputName];
    if (inputDesc && inputDesc.multiArrayConstraint) {
        expectedDataType = inputDesc.multiArrayConstraint.dataType;
    }

    BOOL useCached = (expectedDataType == MLMultiArrayDataTypeFloat16) &&
        [self canUseCachedBufferWithInput:inputName shape:shape outputName:outputName];

    if (useCached) {
        inputArray = _cachedInputArray;
    } else {
        inputArray = [[MLMultiArray alloc]
            initWithShape:shape
                 dataType:expectedDataType
                    error:&error];

        if (error) {
            CACTUS_LOG_ERROR("npu", "ANE create input array failed: " << [[error localizedDescription] UTF8String]);
            return nil;
        }
    }

    NSUInteger totalElements = 1;
    for (NSNumber* dim in shape) {
        totalElements *= [dim unsignedIntegerValue];
    }

    if (expectedDataType == MLMultiArrayDataTypeFloat16) {
        __fp16* inputPtr = (__fp16*)inputArray.dataPointer;
        memcpy(inputPtr, data, totalElements * sizeof(__fp16));
    } else {
        float* inputPtr = (float*)inputArray.dataPointer;
        for (NSUInteger i = 0; i < totalElements; i++) {
            inputPtr[i] = (float)data[i];
        }
    }

    MLFeatureValue* inputFeature = [MLFeatureValue featureValueWithMultiArray:inputArray];
    NSDictionary* inputDict = @{inputName: inputFeature};
    id<MLFeatureProvider> inputProvider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:inputDict
                     error:&error];

    if (error) {
        CACTUS_LOG_ERROR("npu", "ANE create feature provider failed: " << [[error localizedDescription] UTF8String]);
        return nil;
    }

    id<MLFeatureProvider> outputProvider = nil;

    if (useCached && _hasOutputBackings && _predictionOptions) {
        outputProvider = [_model predictionFromFeatures:inputProvider
                                                options:_predictionOptions
                                                  error:&error];
    } else {
        outputProvider = [_model predictionFromFeatures:inputProvider error:&error];
    }

    if (error) {
        CACTUS_LOG_ERROR("npu", "ANE prediction failed: " << [[error localizedDescription] UTF8String]);
        return nil;
    }

    NSString* outName = outputName;
    if (!outName || outName.length == 0) {
        outName = useCached ? _cachedOutputName
                            : _modelDescription.outputDescriptionsByName.allKeys.firstObject;
    }

    MLFeatureValue* outputFeature = [outputProvider featureValueForName:outName];
    static bool logged_output_feature_layout = false;
    if (!logged_output_feature_layout && outputFeature && outputFeature.multiArrayValue) {
        MLMultiArray* outArr = outputFeature.multiArrayValue;
        CACTUS_LOG_DEBUG("npu",
                         "[CactusANE] output feature name="
                             << [outName UTF8String]
                             << " shape=" << [[outArr.shape description] UTF8String]
                             << " strides=" << [[outArr.strides description] UTF8String]
                             << " dtype=" << (long)outArr.dataType);
        logged_output_feature_layout = true;
    }
    return outputFeature.multiArrayValue;
}

static size_t copyMLArrayToFP16(MLMultiArray* array, __fp16* output) {
    size_t count = array.count;
    if (!array || !output || count == 0) return 0;

    std::vector<size_t> dims;
    std::vector<size_t> strides;
    const size_t rank = array.shape.count;
    bool have_layout = rank == array.strides.count && rank > 0;
    if (have_layout) {
        dims.resize(rank);
        strides.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            NSInteger d = [array.shape[i] integerValue];
            NSInteger s = [array.strides[i] integerValue];
            if (d <= 0 || s < 0) {
                have_layout = false;
                break;
            }
            dims[i] = static_cast<size_t>(d);
            strides[i] = static_cast<size_t>(s);
        }
    }

    if (!have_layout) {
        if (array.dataType == MLMultiArrayDataTypeFloat16) {
            if (output != (__fp16*)array.dataPointer) {
                memcpy(output, array.dataPointer, count * sizeof(__fp16));
            }
        } else {
            const float* src = (const float*)array.dataPointer;
            for (size_t i = 0; i < count; i++) output[i] = (__fp16)src[i];
        }
        return count;
    }

    auto is_contiguous = [&]() -> bool {
        if (!have_layout) return false;
        size_t expected = 1;
        for (size_t i = rank; i-- > 0;) {
            if (dims[i] > 1 && strides[i] != expected) return false;
            expected *= dims[i];
        }
        return true;
    };

    if (false && is_contiguous()) {
        if (array.dataType == MLMultiArrayDataTypeFloat16) {
            if (output != (__fp16*)array.dataPointer) {
                memcpy(output, array.dataPointer, count * sizeof(__fp16));
            }
        } else {
            const float* src = (const float*)array.dataPointer;
            for (size_t i = 0; i < count; i++) output[i] = (__fp16)src[i];
        }
        return count;
    }

    std::vector<size_t> idx(rank, 0);
    if (array.dataType == MLMultiArrayDataTypeFloat16) {
        const __fp16* src = (const __fp16*)array.dataPointer;
        for (size_t i = 0; i < count; ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                offset += idx[d] * strides[d];
            }
            output[i] = src[offset];

            for (size_t d = rank; d-- > 0;) {
                idx[d]++;
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
            }
        }
    } else {
        const float* src = (const float*)array.dataPointer;
        for (size_t i = 0; i < count; ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                offset += idx[d] * strides[d];
            }
            output[i] = (__fp16)src[offset];

            for (size_t d = rank; d-- > 0;) {
                idx[d]++;
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
            }
        }
    }
    return count;
}

static void copyFP16ToMLArray(const __fp16* data, size_t count, MLMultiArray* array) {
    if (!array || !data || count == 0) return;

    std::vector<size_t> dims;
    std::vector<size_t> strides;
    const size_t rank = array.shape.count;
    bool have_layout = rank == array.strides.count && rank > 0;
    if (have_layout) {
        dims.resize(rank);
        strides.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            NSInteger d = [array.shape[i] integerValue];
            NSInteger s = [array.strides[i] integerValue];
            if (d <= 0 || s < 0) {
                have_layout = false;
                break;
            }
            dims[i] = static_cast<size_t>(d);
            strides[i] = static_cast<size_t>(s);
        }
    }

    if (!have_layout) {
        if (array.dataType == MLMultiArrayDataTypeFloat16) {
            memcpy(array.dataPointer, data, count * sizeof(__fp16));
        } else {
            float* dst = (float*)array.dataPointer;
            for (size_t i = 0; i < count; i++) dst[i] = (float)data[i];
        }
        return;
    }

    auto is_contiguous = [&]() -> bool {
        if (!have_layout) return false;
        size_t expected = 1;
        for (size_t i = rank; i-- > 0;) {
            if (dims[i] > 1 && strides[i] != expected) return false;
            expected *= dims[i];
        }
        return true;
    };

    if (false && is_contiguous()) {
        if (array.dataType == MLMultiArrayDataTypeFloat16) {
            memcpy(array.dataPointer, data, count * sizeof(__fp16));
        } else {
            float* dst = (float*)array.dataPointer;
            for (size_t i = 0; i < count; i++) dst[i] = (float)data[i];
        }
        return;
    }

    std::vector<size_t> idx(rank, 0);
    if (array.dataType == MLMultiArrayDataTypeFloat16) {
        __fp16* dst = (__fp16*)array.dataPointer;
        for (size_t i = 0; i < count; ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                offset += idx[d] * strides[d];
            }
            dst[offset] = data[i];

            for (size_t d = rank; d-- > 0;) {
                idx[d]++;
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
            }
        }
    } else {
        float* dst = (float*)array.dataPointer;
        for (size_t i = 0; i < count; ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                offset += idx[d] * strides[d];
            }
            dst[offset] = (float)data[i];

            for (size_t d = rank; d-- > 0;) {
                idx[d]++;
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
            }
        }
    }
}

- (BOOL)preallocateMultiInputBuffersWithOutputName:(NSString*)outputName {
    if (!_model || !_modelDescription) return NO;

    NSError* error = nil;
    _cachedMultiInputArrays = [NSMutableDictionary new];

    for (NSString* inputName in _modelDescription.inputDescriptionsByName) {
        MLFeatureDescription* desc = _modelDescription.inputDescriptionsByName[inputName];
        if (desc.type != MLFeatureTypeMultiArray) continue;

        MLMultiArray* array = [[MLMultiArray alloc]
            initWithShape:desc.multiArrayConstraint.shape
                 dataType:desc.multiArrayConstraint.dataType
                    error:&error];
        if (!array || error) return NO;
        _cachedMultiInputArrays[inputName] = array;
    }

    NSString* outName = (outputName && outputName.length > 0)
        ? outputName : _modelDescription.outputDescriptionsByName.allKeys.firstObject;

    MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[outName];
    if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
        _cachedMultiOutputArray = [[MLMultiArray alloc]
            initWithShape:outputDesc.multiArrayConstraint.shape
                 dataType:outputDesc.multiArrayConstraint.dataType
                    error:&error];
        if (!_cachedMultiOutputArray || error) return NO;

        _multiPredictionOptions = [[MLPredictionOptions alloc] init];
        _hasMultiOutputBackings = NO;
    }

    _multiInputPreallocated = YES;
    return YES;
}

- (size_t)predictMultiInputWithDict:(NSMutableDictionary<NSString*, MLFeatureValue*>*)inputDict
                         outputData:(__fp16*)output
                         outputName:(NSString*)outputName {
    if (!_model) return 0;

    NSError* error = nil;
    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:&error];
    if (!provider || error) return 0;

    id<MLFeatureProvider> result = nil;
    if (_hasMultiOutputBackings && _multiPredictionOptions) {
        result = [_model predictionFromFeatures:provider options:_multiPredictionOptions error:&error];
    } else {
        result = [_model predictionFromFeatures:provider error:&error];
    }
    if (!result || error) return 0;

    static bool logged_multi_layout = false;
    if (!logged_multi_layout) {
        for (NSString* key in inputDict) {
            MLFeatureValue* inFeature = inputDict[key];
            if (inFeature && inFeature.multiArrayValue) {
                MLMultiArray* arr = inFeature.multiArrayValue;
            }
        }
    }

    MLFeatureValue* outFeature = [result featureValueForName:outputName];
    if (!outFeature || !outFeature.multiArrayValue) return 0;
    if (!logged_multi_layout) {
        MLMultiArray* outArr = outFeature.multiArrayValue;
        logged_multi_layout = true;
    }
    return copyMLArrayToFP16(outFeature.multiArrayValue, output);
}

@end

namespace cactus {
namespace npu {

static bool g_npu_enabled = true;

ANEEncoder::ANEEncoder() : impl_(nullptr) {}

ANEEncoder::~ANEEncoder() {
    if (impl_) {
        (void)(__bridge_transfer CactusANEImpl*)impl_;
        impl_ = nullptr;
    }
}

ANEEncoder::ANEEncoder(ANEEncoder&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

ANEEncoder& ANEEncoder::operator=(ANEEncoder&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            (void)(__bridge_transfer CactusANEImpl*)impl_;
        }
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool ANEEncoder::load(const std::string& model_path) {
    @autoreleasepool {
        CACTUS_LOG_INFO("npu", "ANEEncoder loading model: " << model_path);
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            CACTUS_LOG_WARN("npu", "not using NPU");
            return false;
        }

        CactusANEImpl* impl = [[CactusANEImpl alloc] initWithModelPath:path];

        if (impl && impl.model) {
            impl_ = (__bridge_retained void*)impl;
            CACTUS_LOG_INFO("npu", "ANEEncoder model loaded successfully: " << model_path);
            return true;
        }
        CACTUS_LOG_ERROR("npu", "ANEEncoder model load failed: " << model_path);
        return false;
    }
}

bool ANEEncoder::preallocate(const std::vector<int>& input_shape,
                             const std::string& input_name,
                             const std::string& output_name) {
    if (!impl_) return false;

    @autoreleasepool {
        CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;

        NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray array];
        for (int dim : input_shape) {
            [shapeArray addObject:@(dim)];
        }

        NSString* inName = [NSString stringWithUTF8String:input_name.c_str()];
        NSString* outName = output_name.empty()
                                ? nil
                                : [NSString stringWithUTF8String:output_name.c_str()];

        return [impl preallocateBuffersWithInput:inName shape:shapeArray outputName:outName];
    }
}

size_t ANEEncoder::encode(const __fp16* input,
                          __fp16* output,
                          const std::vector<int>& shape,
                          const std::string& input_name,
                          const std::string& output_name) {
    if (!impl_ || !input || !output) return 0;

    @autoreleasepool {
        CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;

        NSArray<NSNumber*>* shapeArray = impl.cachedShape;
        bool shapeMatches = (shapeArray && shapeArray.count == shape.size());
        if (shapeMatches) {
            for (size_t i = 0; i < shape.size(); ++i) {
                if ([shapeArray[i] intValue] != shape[i]) {
                    shapeMatches = false;
                    break;
                }
            }
        }

        if (!shapeMatches) {
            NSMutableArray<NSNumber*>* newShapeArray = [NSMutableArray arrayWithCapacity:shape.size()];
            for (int dim : shape) {
                [newShapeArray addObject:@(dim)];
            }
            shapeArray = newShapeArray;
        }

        NSString* inName = impl.cachedInputName;
        if (!inName) {
            inName = [NSString stringWithUTF8String:input_name.c_str()];
        }
        NSString* outName = impl.cachedOutputName;
        if (!outName && !output_name.empty()) {
            outName = [NSString stringWithUTF8String:output_name.c_str()];
        }

        MLMultiArray* mlOutput = [impl predictWithInput:inName
                                                   data:input
                                                  shape:shapeArray
                                             outputName:outName];

        if (mlOutput) {
            return copyMLArrayToFP16(mlOutput, output);
        }
    }

    return 0;
}

bool ANEEncoder::is_available() const {
    if (!impl_) return false;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    return impl.model != nil;
}

std::vector<int> ANEEncoder::get_input_shape() const {
    std::vector<int> result;
    if (!impl_) return result;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    NSArray<NSNumber*>* shape = [impl getInputShape];
    for (NSNumber* dim in shape) {
        result.push_back([dim intValue]);
    }
    return result;
}

std::vector<int> ANEEncoder::get_output_shape() const {
    std::vector<int> result;
    if (!impl_) return result;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    NSArray<NSNumber*>* shape = [impl getOutputShape];
    for (NSNumber* dim in shape) {
        result.push_back([dim intValue]);
    }
    return result;
}

__fp16* ANEEncoder::get_output_buffer() {
    if (!impl_) return nullptr;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    if (!impl.hasOutputBackings || !impl.cachedOutputArray) return nullptr;
    return (__fp16*)impl.cachedOutputArray.dataPointer;
}

size_t ANEEncoder::get_output_buffer_size() const {
    if (!impl_) return 0;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    if (!impl.hasOutputBackings || !impl.cachedOutputArray) return 0;
    return impl.cachedOutputSize;
}

size_t ANEEncoder::encode_multimodal_input(
    const std::vector<NPUNamedInput>& inputs,
    __fp16* output,
    const std::string& output_name) {
    if (!impl_ || !output || inputs.empty()) return 0;

    @autoreleasepool {
        CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;

        NSString* outName = output_name.empty()
            ? impl.modelDescription.outputDescriptionsByName.allKeys.firstObject
            : [NSString stringWithUTF8String:output_name.c_str()];

        if (!impl.multiInputPreallocated) {
            [impl preallocateMultiInputBuffersWithOutputName:outName];
        }

        NSMutableDictionary<NSString*, MLFeatureValue*>* inputDict = [NSMutableDictionary new];

        for (const auto& input : inputs) {
            NSString* nsName = [NSString stringWithUTF8String:input.name.c_str()];

            size_t total = 1;
            for (int dim : input.shape) total *= dim;

            MLMultiArray* array = impl.cachedMultiInputArrays[nsName];
            if (!array) {
                MLMultiArrayDataType dataType = MLMultiArrayDataTypeFloat16;
                MLFeatureDescription* desc = impl.modelDescription.inputDescriptionsByName[nsName];
                if (desc && desc.multiArrayConstraint) dataType = desc.multiArrayConstraint.dataType;

                NSMutableArray<NSNumber*>* shapeArray = [NSMutableArray arrayWithCapacity:input.shape.size()];
                for (int dim : input.shape) [shapeArray addObject:@(dim)];

                NSError* arrayError = nil;
                array = [[MLMultiArray alloc] initWithShape:shapeArray dataType:dataType error:&arrayError];
                if (!array || arrayError) return 0;
            }

            copyFP16ToMLArray(input.data, total, array);
            inputDict[nsName] = [MLFeatureValue featureValueWithMultiArray:array];
        }

        return [impl predictMultiInputWithDict:inputDict outputData:output outputName:outName];
    }
}

std::unique_ptr<NPUEncoder> create_encoder() {
        return std::make_unique<ANEEncoder>();
}

bool is_npu_available() {
    CACTUS_LOG_INFO("npu", "is_npu_available: " << (g_npu_enabled ? "true" : "false"));
    return g_npu_enabled;
}

} // namespace npu
} // namespace cactus

@interface CactusANEPrefillImpl : NSObject

@property (nonatomic, strong) MLModel* model;
@property (nonatomic, strong) MLModelDescription* modelDescription;
@property (nonatomic, assign) int chunkSize;
@property (nonatomic, assign) int hiddenDim;
@property (nonatomic, assign) int numLayers;
@property (nonatomic, assign) int numKvHeads;
@property (nonatomic, assign) int headDim;

@property (nonatomic, strong) MLMultiArray* cachedInputArray;
@property (nonatomic, strong) MLMultiArray* cachedOffsetArray;
@property (nonatomic, strong) NSMutableDictionary<NSString*, MLMultiArray*>* cachedOutputArrays;
@property (nonatomic, strong) MLPredictionOptions* predictionOptions;
@property (nonatomic, assign) BOOL hasOutputBackings;

- (instancetype)initWithModelPath:(NSString*)path;
- (void)preallocateBuffers;
- (BOOL)predictDirectWithInput:(NSString*)inputName
                          data:(const __fp16*)data
                        offset:(int)offset;
- (MLMultiArray*)getOutputArray:(NSString*)name;
@end

@implementation CactusANEPrefillImpl

- (instancetype)initWithModelPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSError* error = nil;
        NSURL* modelURL = resolve_or_compile_model_url(path, &error);

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = resolve_compute_units_for_model(path, true);
        CACTUS_LOG_INFO("npu",
                        "ANEPrefill compute units: "
                            << compute_units_to_string(config.computeUnits)
                            << " for " << [path UTF8String]);

        _model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (_model) {
            _modelDescription = _model.modelDescription;
            [self inferModelDimensions];
            [self preallocateBuffers];
                    }
        if (error) {
            CACTUS_LOG_ERROR("npu", "ANE prefill model load failed: " << [[error localizedDescription] UTF8String]);
        }
    }
    return self;
}

- (void)inferModelDimensions {
    if (!_modelDescription) return;

    NSString* inputName = _modelDescription.inputDescriptionsByName.allKeys.firstObject;
    MLFeatureDescription* inputDesc = _modelDescription.inputDescriptionsByName[inputName];
    if (inputDesc && inputDesc.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* shape = inputDesc.multiArrayConstraint.shape;
        if (shape.count >= 2) {
            _chunkSize = [shape[0] intValue];
            _hiddenDim = [shape[1] intValue];
        }
    }

    int maxLayerIdx = -1;
    for (NSString* outputName in _modelDescription.outputDescriptionsByName.allKeys) {
        if ([outputName hasPrefix:@"k_"]) {
            int layerIdx = [[outputName substringFromIndex:2] intValue];
            maxLayerIdx = MAX(maxLayerIdx, layerIdx);

            MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[outputName];
            if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
                NSArray<NSNumber*>* shape = outputDesc.multiArrayConstraint.shape;
                if (shape.count >= 3) {
                    _numKvHeads = [shape[1] intValue];
                    _headDim = [shape[2] intValue];
                }
            }
        }
    }
    _numLayers = maxLayerIdx + 1;
}

- (void)preallocateBuffers {
    if (!_model || !_modelDescription || _chunkSize == 0 || _hiddenDim == 0) return;

    NSError* error = nil;

    NSArray<NSNumber*>* inputShape = @[@(_chunkSize), @(_hiddenDim)];
    _cachedInputArray = [[MLMultiArray alloc] initWithShape:inputShape
                                                  dataType:MLMultiArrayDataTypeFloat16
                                                     error:&error];
    if (error) {
        _cachedInputArray = nil;
        return;
    }

    if (_modelDescription.inputDescriptionsByName[@"offset"] != nil) {
        _cachedOffsetArray = [[MLMultiArray alloc] initWithShape:@[@1]
                                                       dataType:MLMultiArrayDataTypeFloat16
                                                           error:&error];
        if (error) {
            _cachedOffsetArray = nil;
        }
    }

    _cachedOutputArrays = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString*, MLMultiArray*>* outputBackings = [NSMutableDictionary dictionary];

    for (NSString* outputName in _modelDescription.outputDescriptionsByName.allKeys) {
        MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[outputName];
        if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
            NSArray<NSNumber*>* outputShape = outputDesc.multiArrayConstraint.shape;

            MLMultiArray* outputArray = [[MLMultiArray alloc] initWithShape:outputShape
                                                                  dataType:MLMultiArrayDataTypeFloat16
                                                                     error:&error];
            if (!error && outputArray) {
                _cachedOutputArrays[outputName] = outputArray;
                outputBackings[outputName] = outputArray;
            }
        }
    }

    _predictionOptions = [[MLPredictionOptions alloc] init];
    _hasOutputBackings = NO;
    if (@available(macOS 14.0, iOS 17.0, *)) {
        _predictionOptions.outputBackings = outputBackings;
        _hasOutputBackings = YES;
    }
}

- (BOOL)predictDirectWithInput:(NSString*)inputName
                          data:(const __fp16*)data
                        offset:(int)offset {
    if (!_model || !_cachedInputArray) return NO;

    NSError* error = nil;

    NSUInteger totalElements = (NSUInteger)(_chunkSize * _hiddenDim);
    __fp16* inputPtr = (__fp16*)_cachedInputArray.dataPointer;
    memcpy(inputPtr, data, totalElements * sizeof(__fp16));

    MLFeatureValue* inputFeature = [MLFeatureValue featureValueWithMultiArray:_cachedInputArray];
    NSMutableDictionary* inputDict = [NSMutableDictionary dictionaryWithObject:inputFeature forKey:inputName];

    if (_cachedOffsetArray) {
        ((__fp16*)_cachedOffsetArray.dataPointer)[0] = (__fp16)offset;
        MLFeatureValue* offsetFeature = [MLFeatureValue featureValueWithMultiArray:_cachedOffsetArray];
        inputDict[@"offset"] = offsetFeature;
    }

    id<MLFeatureProvider> inputProvider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:inputDict
                     error:&error];
    if (error) return NO;

    id<MLFeatureProvider> outputProvider = [_model predictionFromFeatures:inputProvider
                                                                  options:_predictionOptions
                                                                    error:&error];
    if (error || !outputProvider) return NO;

    if (!_hasOutputBackings) {
        for (NSString* outputName in _cachedOutputArrays) {
            MLFeatureValue* outFeature = [outputProvider featureValueForName:outputName];
            if (!outFeature || !outFeature.multiArrayValue) return NO;

            MLMultiArray* dst = _cachedOutputArrays[outputName];
            if (!dst) return NO;

            size_t copied = copyMLArrayToFP16(outFeature.multiArrayValue, (__fp16*)dst.dataPointer);
            if (copied == 0) return NO;
        }
    }

    return YES;
}

- (MLMultiArray*)getOutputArray:(NSString*)name {
    return _cachedOutputArrays[name];
}

@end

namespace cactus {
namespace npu {

ANEPrefill::ANEPrefill() : impl_(nullptr) {}

ANEPrefill::~ANEPrefill() {
    if (impl_) {
        (void)(__bridge_transfer CactusANEPrefillImpl*)impl_;
        impl_ = nullptr;
    }
}

ANEPrefill::ANEPrefill(ANEPrefill&& other) noexcept : impl_(other.impl_),
    chunk_size_(other.chunk_size_), hidden_dim_(other.hidden_dim_),
    num_layers_(other.num_layers_), num_kv_heads_(other.num_kv_heads_),
    head_dim_(other.head_dim_) {
    other.impl_ = nullptr;
}

ANEPrefill& ANEPrefill::operator=(ANEPrefill&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            (void)(__bridge_transfer CactusANEPrefillImpl*)impl_;
        }
        impl_ = other.impl_;
        chunk_size_ = other.chunk_size_;
        hidden_dim_ = other.hidden_dim_;
        num_layers_ = other.num_layers_;
        num_kv_heads_ = other.num_kv_heads_;
        head_dim_ = other.head_dim_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool ANEPrefill::load(const std::string& model_path) {
    @autoreleasepool {
        CACTUS_LOG_INFO("npu", "ANEPrefill loading model: " << model_path);
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            CACTUS_LOG_DEBUG("npu", "ANEPrefill model file not found: " << model_path);
            return false;
        }

        CactusANEPrefillImpl* impl = [[CactusANEPrefillImpl alloc] initWithModelPath:path];

        if (impl && impl.model) {
            impl_ = (__bridge_retained void*)impl;
            chunk_size_ = impl.chunkSize;
            hidden_dim_ = impl.hiddenDim;
            num_layers_ = impl.numLayers;
            num_kv_heads_ = impl.numKvHeads;
            head_dim_ = impl.headDim;
            CACTUS_LOG_INFO("npu", "ANEPrefill model loaded successfully: " << model_path);
            return true;
        }
        CACTUS_LOG_ERROR("npu", "ANEPrefill model load failed: " << model_path);
        return false;
    }
}

bool ANEPrefill::is_available() const {
    if (!impl_) return false;
    CactusANEPrefillImpl* impl = (__bridge CactusANEPrefillImpl*)impl_;
    return impl.model != nil;
}

int ANEPrefill::get_chunk_size() const { return chunk_size_; }
int ANEPrefill::get_hidden_dim() const { return hidden_dim_; }
int ANEPrefill::get_num_layers() const { return num_layers_; }
int ANEPrefill::get_num_kv_heads() const { return num_kv_heads_; }
int ANEPrefill::get_head_dim() const { return head_dim_; }

NPUPrefillDirectResult ANEPrefill::prefill_chunk_direct(
    const std::vector<__fp16>& embeddings,
    int position_offset,
    const std::string& input_name) {

    NPUPrefillDirectResult result;
    result.valid = false;
    result.hidden = {nullptr, 0};
    result.k_caches.resize(num_layers_, {nullptr, 0});
    result.v_caches.resize(num_layers_, {nullptr, 0});

    if (!impl_) return result;

    @autoreleasepool {
        CactusANEPrefillImpl* impl = (__bridge CactusANEPrefillImpl*)impl_;

        NSString* inName = [NSString stringWithUTF8String:input_name.c_str()];

        BOOL success = [impl predictDirectWithInput:inName
                                               data:embeddings.data()
                                             offset:position_offset];

        if (!success) return result;

        MLMultiArray* hiddenArray = [impl getOutputArray:@"hidden"];
        if (hiddenArray) {
            result.hidden.data = (__fp16*)hiddenArray.dataPointer;
            result.hidden.count = hiddenArray.count;
        }

        for (int layer = 0; layer < num_layers_; layer++) {
            NSString* kName = [NSString stringWithFormat:@"k_%d", layer];
            NSString* vName = [NSString stringWithFormat:@"v_%d", layer];

            MLMultiArray* kArray = [impl getOutputArray:kName];
            MLMultiArray* vArray = [impl getOutputArray:vName];

            if (kArray) {
                result.k_caches[layer].data = (__fp16*)kArray.dataPointer;
                result.k_caches[layer].count = kArray.count;
            }
            if (vArray) {
                result.v_caches[layer].data = (__fp16*)vArray.dataPointer;
                result.v_caches[layer].count = vArray.count;
            }
        }

        result.valid = true;
    }

    return result;
}

std::unique_ptr<NPUPrefill> create_prefill() {
        return std::make_unique<ANEPrefill>();
}

} // namespace npu
} // namespace cactus

#else // !CACTUS_HAS_ANE

namespace cactus {
namespace npu {

std::unique_ptr<NPUEncoder> create_encoder() {
    return nullptr;
}

bool is_npu_available() {
    return false;
}

std::unique_ptr<NPUPrefill> create_prefill() {
    return nullptr;
}

} // namespace npu
} // namespace cactus

#endif // CACTUS_HAS_ANE
