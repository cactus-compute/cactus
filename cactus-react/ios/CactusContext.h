#ifdef __cplusplus
  #if __has_include(<cactus/llama.h>)
    #import <cactus/llama.h>
    #import <cactus/llama-impl.h>
    #import <cactus/ggml.h>
    #import <cactus/cactus.h>
    #import <cactus/json-schema-to-grammar.h>
  #else
    #import "llama.h"
    #import "llama-impl.h"
    #import "ggml.h"
    #import "cactus.h"
    #import "json-schema-to-grammar.h"
  #endif
#endif


@interface CactusContext : NSObject {
    bool is_metal_enabled;
    bool is_model_loaded;
    NSString * reason_no_metal;

    void (^onProgress)(unsigned int progress);

    cactus::cactus_context * llama;
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog;
+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip;
+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress;
- (void)interruptLoad;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (NSDictionary *)modelInfo;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;
- (void)stopCompletion;
- (NSArray *)tokenize:(NSString *)text;
- (NSArray *)tokenize:(NSString *)text withMediaPaths:(NSArray *)mediaPaths;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params;
- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
    withChatTemplate:(NSString *)chatTemplate
    withJsonSchema:(NSString *)jsonSchema
    withTools:(NSString *)tools
    withParallelToolCalls:(BOOL)parallelToolCalls
    withToolChoice:(NSString *)toolChoice;
- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;
- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr;
- (void)applyLoraAdapters:(NSArray *)loraAdapters;
- (void)removeLoraAdapters;
- (NSArray *)getLoadedLoraAdapters;

// New Multimodal Methods
- (BOOL)initMultimodal:(NSString *)mmprojPath useGpu:(BOOL)useGpu;
- (BOOL)isMultimodalEnabled;
- (BOOL)isMultimodalSupportVision;
- (BOOL)isMultimodalSupportAudio;
- (void)releaseMultimodal;
- (NSDictionary *)multimodalCompletion:(NSString *)prompt 
    withMediaPaths:(NSArray *)mediaPaths 
    params:(NSDictionary *)params 
    onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;

// New TTS/Vocoder Methods
- (BOOL)initVocoder:(NSString *)vocoderModelPath;
- (BOOL)isVocoderEnabled;
- (int)getTTSType;
- (NSString *)getFormattedAudioCompletion:(NSString *)speakerJsonStr textToSpeak:(NSString *)textToSpeak;
- (NSArray *)getAudioCompletionGuideTokens:(NSString *)textToSpeak;
- (NSArray *)decodeAudioTokens:(NSArray *)tokens;
- (void)releaseVocoder;

- (void)invalidate;

@end
