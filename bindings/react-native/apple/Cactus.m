#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(Cactus, RCTEventEmitter)

RCT_EXTERN_METHOD(init:(NSString *)modelPath corpusDir:(NSString * _Nullable)corpusDir cacheIndex:(BOOL)cacheIndex resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(destroy:(NSString *)handle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(reset:(NSString *)handle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(stop:(NSString *)handle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(prefill:(NSString *)handle messagesJson:(NSString *)messagesJson optionsJson:(NSString * _Nullable)optionsJson toolsJson:(NSString * _Nullable)toolsJson pcmDataBase64:(NSString * _Nullable)pcmDataBase64 resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(complete:(NSString *)handle messagesJson:(NSString *)messagesJson optionsJson:(NSString * _Nullable)optionsJson toolsJson:(NSString * _Nullable)toolsJson pcmDataBase64:(NSString * _Nullable)pcmDataBase64 streamTokens:(BOOL)streamTokens resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(tokenize:(NSString *)handle text:(NSString *)text resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(scoreWindow:(NSString *)handle tokens:(NSArray<NSNumber *> *)tokens start:(nonnull NSNumber *)start end:(nonnull NSNumber *)end context:(nonnull NSNumber *)context resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(transcribe:(NSString *)handle audioPath:(NSString * _Nullable)audioPath prompt:(NSString * _Nullable)prompt optionsJson:(NSString * _Nullable)optionsJson pcmDataBase64:(NSString * _Nullable)pcmDataBase64 resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(streamTranscribeStart:(NSString *)handle optionsJson:(NSString * _Nullable)optionsJson resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(streamTranscribeProcess:(NSString *)streamHandle pcmDataBase64:(NSString * _Nullable)pcmDataBase64 resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(streamTranscribeStop:(NSString *)streamHandle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(embed:(NSString *)handle text:(NSString *)text normalize:(BOOL)normalize resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(imageEmbed:(NSString *)handle imagePath:(NSString *)imagePath resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(audioEmbed:(NSString *)handle audioPath:(NSString *)audioPath resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(ragQuery:(NSString *)handle query:(NSString *)query topK:(nonnull NSNumber *)topK resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexInit:(NSString *)indexDir embeddingDim:(nonnull NSNumber *)embeddingDim resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexAdd:(NSString *)handle ids:(NSArray<NSNumber *> *)ids documents:(NSArray<NSString *> *)documents embeddings:(NSArray<NSArray<NSNumber *> *> *)embeddings metadatas:(NSArray<NSString *> * _Nullable)metadatas resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexDelete:(NSString *)handle ids:(NSArray<NSNumber *> *)ids resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexGet:(NSString *)handle ids:(NSArray<NSNumber *> *)ids resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexQuery:(NSString *)handle embedding:(NSArray<NSNumber *> *)embedding optionsJson:(NSString * _Nullable)optionsJson resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexCompact:(NSString *)handle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(indexDestroy:(NSString *)handle resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(logSetLevel:(nonnull NSNumber *)level resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(setTelemetryEnvironment:(NSString * _Nullable)framework cacheLocation:(NSString * _Nullable)cacheLocation version:(NSString * _Nullable)version resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(setAppId:(NSString *)appId resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(telemetryFlush:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(telemetryShutdown:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(getLastError:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

@end
