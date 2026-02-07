#include "npu_ane.h"

#if CACTUS_HAS_ANE

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include "../graph/graph.h"

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

@end

@implementation CactusANEImpl

- (instancetype)initWithModelPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSError* error = nil;
        NSURL* modelURL = [NSURL fileURLWithPath:path];

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        if ([path hasSuffix:@".mlpackage"]) {
            BOOL isDir = NO;
            if (![[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir] || !isDir) {
                CACTUS_LOG_ERROR("npu", "ANE mlpackage path is not a valid directory: " << [path UTF8String]);
                return self;
            }
            NSString* cachedPath = [[path stringByDeletingPathExtension] stringByAppendingPathExtension:@"mlmodelc"];
            NSURL* cachedURL = [NSURL fileURLWithPath:cachedPath];

            if ([[NSFileManager defaultManager] fileExistsAtPath:cachedPath]) {
                modelURL = cachedURL;
            } else {
                NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
                if (error) {
                    CACTUS_LOG_ERROR("npu", "ANE model compilation failed: " << [[error localizedDescription] UTF8String]);
                    return self;
                }

                NSError* moveError = nil;
                [[NSFileManager defaultManager] moveItemAtURL:compiledURL toURL:cachedURL error:&moveError];
                if (moveError) {
                    modelURL = compiledURL;
                } else {
                    modelURL = cachedURL;
                }
            }
        }

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

    _cachedInputArray = [[MLMultiArray alloc]
        initWithShape:shape
             dataType:MLMultiArrayDataTypeFloat16
                error:&error];

    if (error) {
        CACTUS_LOG_ERROR("npu", "ANE preallocate input array failed: " << [[error localizedDescription] UTF8String]);
        return NO;
    }

    _cachedShape = [shape copy];
    _cachedInputName = [inputName copy];
    _cachedOutputName = outputName ? [outputName copy]
                                   : _modelDescription.outputDescriptionsByName.allKeys.firstObject;

    MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[_cachedOutputName];
    if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* outputShape = outputDesc.multiArrayConstraint.shape;

        _cachedOutputArray = [[MLMultiArray alloc]
            initWithShape:outputShape
                 dataType:MLMultiArrayDataTypeFloat16
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

        _predictionOptions = [[MLPredictionOptions alloc] init];
        if (@available(macOS 14.0, iOS 17.0, *)) {
            _predictionOptions.outputBackings = @{_cachedOutputName: _cachedOutputArray};
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

    BOOL useCached = [self canUseCachedBufferWithInput:inputName shape:shape outputName:outputName];

    if (useCached) {
        inputArray = _cachedInputArray;
    } else {
        inputArray = [[MLMultiArray alloc]
            initWithShape:shape
                 dataType:MLMultiArrayDataTypeFloat16
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

    __fp16* inputPtr = (__fp16*)inputArray.dataPointer;
    memcpy(inputPtr, data, totalElements * sizeof(__fp16));

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

    if (useCached && _predictionOptions) {
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

    if (useCached && _predictionOptions && _cachedOutputArray) {
        return _cachedOutputArray;
    }

    MLFeatureValue* outputFeature = [outputProvider featureValueForName:outName];
    return outputFeature.multiArrayValue;
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
            CACTUS_LOG_ERROR("npu", "ANEEncoder model file not found: " << model_path);
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

        // Use cached names
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
            size_t count = mlOutput.count;
            __fp16* outputPtr = (__fp16*)mlOutput.dataPointer;
            if (output != outputPtr) {
                memcpy(output, outputPtr, count * sizeof(__fp16));
            }
            return count;
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
    if (!impl.cachedOutputArray) return nullptr;
    return (__fp16*)impl.cachedOutputArray.dataPointer;
}

size_t ANEEncoder::get_output_buffer_size() const {
    if (!impl_) return 0;
    CactusANEImpl* impl = (__bridge CactusANEImpl*)impl_;
    return impl.cachedOutputSize;
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
        NSURL* modelURL = [NSURL fileURLWithPath:path];

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        if ([path hasSuffix:@".mlpackage"]) {
            BOOL isDir = NO;
            if (![[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir] || !isDir) {
                CACTUS_LOG_ERROR("npu", "ANE prefill mlpackage path is not a valid directory: " << [path UTF8String]);
                return self;
            }
            NSString* cachedPath = [[path stringByDeletingPathExtension] stringByAppendingPathExtension:@"mlmodelc"];
            NSURL* cachedURL = [NSURL fileURLWithPath:cachedPath];

            if ([[NSFileManager defaultManager] fileExistsAtPath:cachedPath]) {
                modelURL = cachedURL;
            } else {
                NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
                if (error) {
                    CACTUS_LOG_ERROR("npu", "ANE prefill model compilation failed: " << [[error localizedDescription] UTF8String]);
                    return self;
                }

                NSError* moveError = nil;
                [[NSFileManager defaultManager] moveItemAtURL:compiledURL toURL:cachedURL error:&moveError];
                if (moveError) {
                    modelURL = compiledURL;
                } else {
                    modelURL = cachedURL;
                }
            }
        }

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
    if (@available(macOS 14.0, iOS 17.0, *)) {
        _predictionOptions.outputBackings = outputBackings;
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
    return (error == nil && outputProvider != nil);
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
            CACTUS_LOG_ERROR("npu", "ANEPrefill model file not found: " << model_path);
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