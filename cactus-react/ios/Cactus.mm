#import "Cactus.h"
#import "CactusContext.h"
#include "cactus_ffi.h" // For STT FFI functions

#ifdef RCT_NEW_ARCH_ENABLED
#import "CactusSpec.h"
#endif

// STT Context Management - simple global for now, like in JNI.
// Proper multi-instance would need a dictionary or similar.
static cactus_stt_context_t* g_stt_context_ios = nullptr;

// C-style functions to be called from Swift via Bridging Header
extern "C" {
    void* RN_STT_init(const char* model_path, const char* language) {
        if (g_stt_context_ios != nullptr) {
            cactus_stt_free(g_stt_context_ios);
            g_stt_context_ios = nullptr;
        }
        g_stt_context_ios = cactus_stt_init(model_path, language);
        return g_stt_context_ios; // Return as void* to Swift
    }

    void RN_STT_free(void* stt_context_ptr) {
        if (stt_context_ptr && stt_context_ptr == g_stt_context_ios) {
            cactus_stt_free(g_stt_context_ios);
            g_stt_context_ios = nullptr;
        }
    }

    void RN_STT_setUserVocabulary(void* stt_context_ptr, const char* vocabulary) {
        if (!stt_context_ptr || stt_context_ptr != g_stt_context_ios) {
            // Optional: Log error if context is null or doesn't match global
            return;
        }
        cactus_stt_set_user_vocabulary(g_stt_context_ios, vocabulary);
    }

    // Placeholder for processAudioFile - Swift side will handle file reading if needed,
    // or this would take raw audio samples if design changes.
    // For now, let's assume it takes a path and returns a transcription string.
    const char* RN_STT_processAudioFile(void* stt_context_ptr, const char* file_path) {
        if (!stt_context_ptr || stt_context_ptr != g_stt_context_ios || !file_path) {
            return nullptr;
        }
        // This is a simplified placeholder. Actual processing would involve:
        // 1. Reading audio file into float samples (not done here).
        // 2. Calling cactus_stt_process_audio().
        // 3. Calling cactus_stt_get_transcription().
        // For now, just return a placeholder or empty string if context is valid.
        // This example returns a new string that Swift side would need to free.
        // Or, modify to return a static string for placeholder.
        // Let's return what getTranscription would return (might be empty if no audio processed)
        const char* transcription = cactus_stt_get_transcription(g_stt_context_ios);
        // The FFI `cactus_stt_get_transcription` returns a string that must be freed by `cactus_free_string_c`.
        // The Swift side will be responsible for this.
        return transcription;
    }

    void RN_STT_free_string(char* str_ptr) {
        if (str_ptr) {
            cactus_free_string_c(str_ptr);
        }
    }
} // extern "C"


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
    NSMutableArray *tokens = [context tokenize:text];
    resolve(@{ @"tokens": tokens });
    [tokens release];
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
