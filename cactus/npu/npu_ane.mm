#include "npu_ane.h"

#if CACTUS_HAS_ANE

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <iostream>

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
        NSURL* modelURL = [NSURL fileURLWithPath:path];
        NSError* error = nil;

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        _model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (_model) {
            _modelDescription = _model.modelDescription;
        }
        if (error) {
            NSLog(@"[CactusANE] Error loading model: %@", error);
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
        NSLog(@"[CactusANE] Error preallocating input array: %@", error);
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
            NSLog(@"[CactusANE] Error preallocating output array: %@", error);
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
            NSLog(@"[CactusANE] Error creating input array: %@", error);
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
        NSLog(@"[CactusANE] Error creating feature provider: %@", error);
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
        NSLog(@"[CactusANE] Error during prediction: %@", error);
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
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            return false;
        }

        CactusANEImpl* impl = [[CactusANEImpl alloc] initWithModelPath:path];

        if (impl && impl.model) {
            impl_ = (__bridge_retained void*)impl;
            return true;
        }
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
    return true;
}

} // namespace npu
} // namespace cactus

// ============================================================================
// CactusANEPrefillImpl - Objective-C implementation for multi-output prefill
// Must be at global scope, not inside C++ namespace
// ============================================================================

@interface CactusANEPrefillImpl : NSObject

@property (nonatomic, strong) MLModel* model;
@property (nonatomic, strong) MLModelDescription* modelDescription;
@property (nonatomic, assign) int chunkSize;
@property (nonatomic, assign) int hiddenDim;
@property (nonatomic, assign) int numLayers;
@property (nonatomic, assign) int numKvHeads;
@property (nonatomic, assign) int headDim;

- (instancetype)initWithModelPath:(NSString*)path;
- (NSArray<NSDictionary*>*)predictWithInput:(NSString*)inputName
                                       data:(const __fp16*)data
                                      shape:(NSArray<NSNumber*>*)shape
                                     offset:(int)offset;

@end

@implementation CactusANEPrefillImpl

- (instancetype)initWithModelPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSURL* modelURL = [NSURL fileURLWithPath:path];
        NSError* error = nil;

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        _model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (_model) {
            _modelDescription = _model.modelDescription;
            [self inferModelDimensions];
        }
        if (error) {
            NSLog(@"[CactusANEPrefill] Error loading model: %@", error);
        }
    }
    return self;
}

- (void)inferModelDimensions {
    if (!_modelDescription) return;

    // Get input shape to determine chunk_size and hidden_dim
    // Expected input: [chunk_size, hidden_dim] e.g., [256, 1024]
    NSString* inputName = _modelDescription.inputDescriptionsByName.allKeys.firstObject;
    MLFeatureDescription* inputDesc = _modelDescription.inputDescriptionsByName[inputName];
    if (inputDesc && inputDesc.type == MLFeatureTypeMultiArray) {
        NSArray<NSNumber*>* shape = inputDesc.multiArrayConstraint.shape;
        if (shape.count >= 2) {
            _chunkSize = [shape[0] intValue];
            _hiddenDim = [shape[1] intValue];
        }
    }

    // Infer num_layers, num_kv_heads, head_dim from output names
    // Outputs are: hidden, k_0, v_0, k_1, v_1, ..., k_N, v_N
    // KV shape: [chunk_size, num_kv_heads, head_dim]
    int maxLayerIdx = -1;
    for (NSString* outputName in _modelDescription.outputDescriptionsByName.allKeys) {
        if ([outputName hasPrefix:@"k_"]) {
            int layerIdx = [[outputName substringFromIndex:2] intValue];
            maxLayerIdx = MAX(maxLayerIdx, layerIdx);

            // Get KV dimensions from this output
            MLFeatureDescription* outputDesc = _modelDescription.outputDescriptionsByName[outputName];
            if (outputDesc && outputDesc.type == MLFeatureTypeMultiArray) {
                NSArray<NSNumber*>* shape = outputDesc.multiArrayConstraint.shape;
                // Shape: [chunk_size, num_kv_heads, head_dim]
                if (shape.count >= 3) {
                    _numKvHeads = [shape[1] intValue];
                    _headDim = [shape[2] intValue];
                }
            }
        }
    }
    _numLayers = maxLayerIdx + 1;

    NSLog(@"[CactusANEPrefill] Model dimensions: chunk_size=%d, hidden_dim=%d, layers=%d, kv_heads=%d, head_dim=%d",
          _chunkSize, _hiddenDim, _numLayers, _numKvHeads, _headDim);
}

- (NSArray<NSDictionary*>*)predictWithInput:(NSString*)inputName
                                       data:(const __fp16*)data
                                      shape:(NSArray<NSNumber*>*)shape
                                     offset:(int)offset {
    if (!_model) return @[];

    NSError* error = nil;

    // Create input array for embeddings
    MLMultiArray* inputArray = [[MLMultiArray alloc]
        initWithShape:shape
             dataType:MLMultiArrayDataTypeFloat16
                error:&error];

    if (error) {
        NSLog(@"[CactusANEPrefill] Error creating input array: %@", error);
        return @[];
    }

    // Copy input data
    NSUInteger totalElements = 1;
    for (NSNumber* dim in shape) {
        totalElements *= [dim unsignedIntegerValue];
    }
    __fp16* inputPtr = (__fp16*)inputArray.dataPointer;
    memcpy(inputPtr, data, totalElements * sizeof(__fp16));

    // Create feature provider with embeddings
    MLFeatureValue* inputFeature = [MLFeatureValue featureValueWithMultiArray:inputArray];
    NSMutableDictionary* inputDict = [NSMutableDictionary dictionaryWithObject:inputFeature forKey:inputName];

    // Add offset input if model supports it (for RoPE position encoding)
    if (_modelDescription.inputDescriptionsByName[@"offset"] != nil) {
        MLMultiArray* offsetArray = [[MLMultiArray alloc] initWithShape:@[@1]
                                                               dataType:MLMultiArrayDataTypeInt32
                                                                  error:&error];
        if (!error) {
            ((int32_t*)offsetArray.dataPointer)[0] = offset;
            MLFeatureValue* offsetFeature = [MLFeatureValue featureValueWithMultiArray:offsetArray];
            inputDict[@"offset"] = offsetFeature;
        }
    }

    id<MLFeatureProvider> inputProvider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:inputDict
                     error:&error];

    if (error) {
        NSLog(@"[CactusANEPrefill] Error creating feature provider: %@", error);
        return @[];
    }

    // Run prediction
    id<MLFeatureProvider> outputProvider = [_model predictionFromFeatures:inputProvider error:&error];

    if (error) {
        NSLog(@"[CactusANEPrefill] Error during prediction: %@", error);
        return @[];
    }

    // Collect all outputs
    NSMutableArray<NSDictionary*>* results = [NSMutableArray array];
    for (NSString* outputName in _modelDescription.outputDescriptionsByName.allKeys) {
        MLFeatureValue* outputFeature = [outputProvider featureValueForName:outputName];
        if (outputFeature && outputFeature.multiArrayValue) {
            MLMultiArray* outputArray = outputFeature.multiArrayValue;

            // Get shape
            NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
            for (NSNumber* dim in outputArray.shape) {
                [outputShape addObject:dim];
            }

            // Copy data
            size_t count = outputArray.count;
            NSMutableData* outputData = [NSMutableData dataWithLength:count * sizeof(__fp16)];
            __fp16* outputPtr = (__fp16*)outputArray.dataPointer;
            memcpy(outputData.mutableBytes, outputPtr, count * sizeof(__fp16));

            [results addObject:@{
                @"name": outputName,
                @"shape": outputShape,
                @"data": outputData
            }];
        }
    }

    return results;
}

@end

// C++ wrapper implementation for ANEPrefill
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
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
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
            return true;
        }
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

std::vector<NPUPrefillOutput> ANEPrefill::prefill_chunk(
    const std::vector<__fp16>& embeddings,
    int position_offset,
    const std::string& input_name) {

    std::vector<NPUPrefillOutput> results;
    if (!impl_) return results;

    @autoreleasepool {
        CactusANEPrefillImpl* impl = (__bridge CactusANEPrefillImpl*)impl_;

        NSString* inName = [NSString stringWithUTF8String:input_name.c_str()];
        NSArray<NSNumber*>* shape = @[@(chunk_size_), @(hidden_dim_)];

        NSArray<NSDictionary*>* outputs = [impl predictWithInput:inName
                                                            data:embeddings.data()
                                                           shape:shape
                                                          offset:position_offset];

        for (NSDictionary* output in outputs) {
            NPUPrefillOutput result;
            result.name = [output[@"name"] UTF8String];

            NSArray<NSNumber*>* shapeArray = output[@"shape"];
            for (NSNumber* dim in shapeArray) {
                result.shape.push_back([dim intValue]);
            }

            NSData* data = output[@"data"];
            size_t count = data.length / sizeof(__fp16);
            result.data.resize(count);
            memcpy(result.data.data(), data.bytes, data.length);

            results.push_back(std::move(result));
        }
    }

    return results;
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