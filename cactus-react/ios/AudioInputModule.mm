// AudioInputModule.mm
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

// It's crucial that the project's bridging header is correctly set up in build settings
// for the `-Swift.h` header to be generated and found.
// This generated header makes Swift classes visible to Objective-C.
// The name of this header is typically "ProjectName-Swift.h".
// Assuming the project name derived from the folder structure might be "CactusReact":
#import "CactusReact-Swift.h"


// Register the Swift module with React Native
// The first argument is the Swift class name.
// The second argument is the superclass, which is RCTEventEmitter if your Swift class inherits from it.
// If AudioInputModule is a direct subclass of NSObject (and not RCTEventEmitter directly in Swift,
// but conforms to RCTBridgeModule), then you might not specify a superclass here or use NSObject.
// However, since AudioInputModule is an RCTEventEmitter, we specify it.
RCT_EXTERN_MODULE(AudioInputModule, RCTEventEmitter)

// Expose the requestPermissions method to JavaScript
// RCT_EXTERN_METHOD(methodName:(paramType)paramName ... resolve:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
RCT_EXTERN_METHOD(requestPermissions:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

// Expose the startRecording method to JavaScript
RCT_EXTERN_METHOD(startRecording:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

// Expose the stopRecording method to JavaScript
RCT_EXTERN_METHOD(stopRecording:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

// Expose STT methods to JavaScript
RCT_EXTERN_METHOD(initSTT:(NSString *)modelPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(processAudioFile:(NSString *)filePath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(releaseSTT:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

// If you had synchronous methods, they would be exposed like this:
// RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(someSynchronousMethod)
// However, all current methods are asynchronous and return Promises.

// No additional implementation is typically needed in this .mm file for simple proxying
// of methods and events. React Native's macros and the Swift class itself handle the logic.
// Event emissions are handled by AudioInputModule (as an RCTEventEmitter) directly.
