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

#else // !CACTUS_HAS_ANE

namespace cactus {
namespace npu {

std::unique_ptr<NPUEncoder> create_encoder() {
    return nullptr;
}

bool is_npu_available() {
    return false;
}

} // namespace npu
} // namespace cactus

#endif // CACTUS_HAS_ANE