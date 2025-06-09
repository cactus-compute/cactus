#import "Cactus.h"
#import "CactusContext.h"

#ifdef RCT_NEW_ARCH_ENABLED
#import "CactusSpec.h"
#endif

@implementation Cactus

NSMutableDictionary *llamaContexts;
double llamaContextLimit = -1;
dispatch_queue_t llamaDQueue;

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(toggleNativeLog:(BOOL)enabled) {
    void (^onEmitLog)(NSString *level, NSString *text) = nil;
    if (enabled) {
        onEmitLog = ^(NSString *level, NSString *text) {
            [self sendEventWithName:@"@Cactus_onNativeLog" body:@{ @"level": level, @"text": text }];
        };
    }
    [CactusContext toggleNativeLog:enabled onEmitLog:onEmitLog];
}

RCT_EXPORT_METHOD(setContextLimit:(double)limit
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    llamaContextLimit = limit;
    resolve(nil);
}

RCT_EXPORT_METHOD(modelInfo:(NSString *)path
                 withSkip:(NSArray *)skip
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    resolve([CactusContext modelInfo:path skip:skip]);
}

RCT_EXPORT_METHOD(initContext:(double)contextId
                 withContextParams:(NSDictionary *)contextParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSNumber *contextIdNumber = [NSNumber numberWithDouble:contextId];
    if (llamaContexts[contextIdNumber] != nil) {
        reject(@"llama_error", @"Context already exists", nil);
        return;
    }

    if (llamaDQueue == nil) {
        llamaDQueue = dispatch_queue_create("com.cactus", DISPATCH_QUEUE_SERIAL);
    }

    if (llamaContexts == nil) {
        llamaContexts = [[NSMutableDictionary alloc] init];
    }

    if (llamaContextLimit > -1 && [llamaContexts count] >= llamaContextLimit) {
        reject(@"llama_error", @"Context limit reached", nil);
        return;
    }

    @try {
      CactusContext *context = [CactusContext initWithParams:contextParams onProgress:^(unsigned int progress) {
          dispatch_async(dispatch_get_main_queue(), ^{
              [self sendEventWithName:@"@Cactus_onInitContextProgress" body:@{ @"contextId": @(contextId), @"progress": @(progress) }];
          });
      }];
      if (![context isModelLoaded]) {
          reject(@"llama_cpp_error", @"Failed to load the model", nil);
          return;
      }

      [llamaContexts setObject:context forKey:contextIdNumber];

      resolve(@{
          @"gpu": @([context isMetalEnabled]),
          @"reasonNoGPU": [context reasonNoMetal],
          @"model": [context modelInfo],
      });
    } @catch (NSException *exception) {
      reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(getFormattedChat:(double)contextId
                 withMessages:(NSString *)messages
                 withTemplate:(NSString *)chatTemplate
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    try {
        if ([params[@"jinja"] boolValue]) {
            NSString *jsonSchema = params[@"json_schema"];
            NSString *tools = params[@"tools"];
            bool parallelToolCalls = [params[@"parallel_tool_calls"] boolValue];
            NSString *toolChoice = params[@"tool_choice"];
            resolve([context getFormattedChatWithJinja:messages withChatTemplate:chatTemplate withJsonSchema:jsonSchema withTools:tools withParallelToolCalls:parallelToolCalls withToolChoice:toolChoice]);
        } else {
            resolve([context getFormattedChat:messages withChatTemplate:chatTemplate]);
        }
    } catch (const std::exception& e) { // catch cpp exceptions
        reject(@"llama_error", [NSString stringWithUTF8String:e.what()], nil);
    }
}

RCT_EXPORT_METHOD(loadSession:(double)contextId
                 withFilePath:(NSString *)filePath
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                resolve([context loadSession:filePath]);
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(saveSession:(double)contextId
                 withFilePath:(NSString *)filePath
                 withSize:(double)size
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                int count = [context saveSession:filePath size:(int)size];
                resolve(@(count));
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
        }
    });
}

- (NSArray *)supportedEvents {
  return@[
    @"@Cactus_onInitContextProgress",
    @"@Cactus_onToken",
    @"@Cactus_onNativeLog",
  ];
}

RCT_EXPORT_METHOD(completion:(double)contextId
                 withCompletionParams:(NSDictionary *)completionParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                NSDictionary* completionResult = [context completion:completionParams
                    onToken:^(NSMutableDictionary *tokenResult) {
                        if (![completionParams[@"emit_partial_completion"] boolValue]) return;
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [self sendEventWithName:@"@Cactus_onToken"
                                body:@{
                                    @"contextId": [NSNumber numberWithDouble:contextId],
                                    @"tokenResult": tokenResult
                                }
                            ];
                            [tokenResult release];
                        });
                    }
                ];
                resolve(completionResult);
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
            [context stopCompletion];
        }
    });

}

RCT_EXPORT_METHOD(stopCompletion:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    [context stopCompletion];
    resolve(nil);
}

RCT_EXPORT_METHOD(tokenize:(double)contextId
                  text:(NSString *)text
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    NSArray *tokens = [context tokenize:text];
    resolve(@{ @"tokens": tokens });
}

RCT_EXPORT_METHOD(tokenize:(double)contextId
                  text:(NSString *)text
                  mediaPaths:(NSArray *)mediaPaths
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    
    if (mediaPaths && [mediaPaths count] > 0) {
        resolve([context tokenize:text withMediaPaths:mediaPaths]);
    } else {
        NSArray *tokens = [context tokenize:text];
        resolve(@{ @"tokens": tokens });
    }
}

RCT_EXPORT_METHOD(detokenize:(double)contextId
                  tokens:(NSArray *)tokens
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve([context detokenize:tokens]);
}

RCT_EXPORT_METHOD(embedding:(double)contextId
                  text:(NSString *)text
                  params:(NSDictionary *)params
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSDictionary *embedding = [context embedding:text params:params];
        resolve(embedding);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(bench:(double)contextId
                  pp:(int)pp
                  tg:(int)tg
                  pl:(int)pl
                  nr:(int)nr
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSString *benchResults = [context bench:pp tg:tg pl:pl nr:nr];
        resolve(benchResults);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(applyLoraAdapters:(double)contextId
                 withLoraAdapters:(NSArray *)loraAdapters
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    [context applyLoraAdapters:loraAdapters];
    resolve(nil);
}

RCT_EXPORT_METHOD(removeLoraAdapters:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    [context removeLoraAdapters];
    resolve(nil);
}

RCT_EXPORT_METHOD(getLoadedLoraAdapters:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve([context getLoadedLoraAdapters]);
}

RCT_EXPORT_METHOD(initMultimodal:(double)contextId
                 mmprojPath:(NSString *)mmprojPath
                 useGpu:(BOOL)useGpu
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        BOOL result = [context initMultimodal:mmprojPath useGpu:useGpu];
        resolve(@(result));
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(isMultimodalEnabled:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve(@([context isMultimodalEnabled]));
}

RCT_EXPORT_METHOD(isMultimodalSupportVision:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve(@([context isMultimodalSupportVision]));
}

RCT_EXPORT_METHOD(isMultimodalSupportAudio:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve(@([context isMultimodalSupportAudio]));
}

RCT_EXPORT_METHOD(releaseMultimodal:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    [context releaseMultimodal];
    resolve(nil);
}

RCT_EXPORT_METHOD(multimodalCompletion:(double)contextId
                 prompt:(NSString *)prompt
                 mediaPaths:(NSArray *)mediaPaths
                 completionParams:(NSDictionary *)completionParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                NSDictionary* completionResult = [context multimodalCompletion:prompt
                    withMediaPaths:mediaPaths
                    params:completionParams
                    onToken:^(NSMutableDictionary *tokenResult) {
                        if (![completionParams[@"emit_partial_completion"] boolValue]) return;
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [self sendEventWithName:@"@Cactus_onToken"
                                body:@{
                                    @"contextId": [NSNumber numberWithDouble:contextId],
                                    @"tokenResult": tokenResult
                                }
                            ];
                            [tokenResult release];
                        });
                    }
                ];
                resolve(completionResult);
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
            [context stopCompletion];
        }
    });
}

RCT_EXPORT_METHOD(releaseContext:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if (![context isModelLoaded]) {
      [context interruptLoad];
    }
    [context stopCompletion];
    dispatch_barrier_sync(llamaDQueue, ^{});
    [context invalidate];
    [llamaContexts removeObjectForKey:[NSNumber numberWithDouble:contextId]];
    resolve(nil);
}

RCT_EXPORT_METHOD(releaseAllContexts:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    [self invalidate];
    resolve(nil);
}

// New TTS/Vocoder Methods
RCT_EXPORT_METHOD(initVocoder:(double)contextId
                 vocoderModelPath:(NSString *)vocoderModelPath
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        BOOL result = [context initVocoder:vocoderModelPath];
        resolve(@(result));
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(isVocoderEnabled:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve(@([context isVocoderEnabled]));
}

RCT_EXPORT_METHOD(getTTSType:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve(@{ @"type": @([context getTTSType]) });
}

RCT_EXPORT_METHOD(getFormattedAudioCompletion:(double)contextId
                 speakerJsonStr:(NSString *)speakerJsonStr
                 textToSpeak:(NSString *)textToSpeak
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSString *result = [context getFormattedAudioCompletion:speakerJsonStr textToSpeak:textToSpeak];
        resolve(@{ @"formatted_prompt": result });
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(getAudioCompletionGuideTokens:(double)contextId
                 textToSpeak:(NSString *)textToSpeak
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSArray *tokens = [context getAudioCompletionGuideTokens:textToSpeak];
        resolve(@{ @"tokens": tokens });
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(decodeAudioTokens:(double)contextId
                 tokens:(NSArray *)tokens
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSArray *audioData = [context decodeAudioTokens:tokens];
        resolve(@{ @"audio_data": audioData });
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(releaseVocoder:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    CactusContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    [context releaseVocoder];
    resolve(nil);
}

- (void)invalidate {
    if (llamaContexts == nil) {
        return;
    }

    for (NSNumber *contextId in llamaContexts) {
        CactusContext *context = llamaContexts[contextId];
        [context stopCompletion];
        dispatch_barrier_sync(llamaDQueue, ^{});
        [context invalidate];
    }

    [llamaContexts removeAllObjects];
    [llamaContexts release];
    llamaContexts = nil;

    if (llamaDQueue != nil) {
        dispatch_release(llamaDQueue);
        llamaDQueue = nil;
    }

    [super invalidate];
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeCactusSpecJSI>(params);
}
#endif

@end
