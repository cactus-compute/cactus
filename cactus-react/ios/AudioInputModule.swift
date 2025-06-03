// AudioInputModule.swift
import Foundation
import AVFoundation

@objc(AudioInputModule)
class AudioInputModule: RCTEventEmitter {

  private var audioRecorder: AVAudioRecorder?
  private var audioFilename: URL?
  private var sttContext: UnsafeMutableRawPointer? // To store the cactus_stt_context_t*

  override init() {
    super.init()
  }

  @objc
  func requestPermissions(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      if granted {
        resolve(true)
      } else {
        reject("permission_denied", "Microphone permission denied", nil)
      }
    }
  }

  @objc
  func startRecording(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    let audioSession = AVAudioSession.sharedInstance()
    do {
      try audioSession.setCategory(.playAndRecord, mode: .default)
      try audioSession.setActive(true)

      let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      self.audioFilename = documentsPath.appendingPathComponent("recording.m4a")

      guard let audioFilename = self.audioFilename else {
        reject("file_error", "Could not create audio file", nil)
        return
      }

      let settings = [
        AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
        AVSampleRateKey: 12000,
        AVNumberOfChannelsKey: 1,
        AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
      ]

      self.audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
      self.audioRecorder?.delegate = self
      self.audioRecorder?.record()
      resolve("Recording started")
    } catch {
      reject("start_recording_failed", "Failed to start recording: \(error.localizedDescription)", error)
    }
  }

  @objc
  func stopRecording(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let recorder = self.audioRecorder else {
      reject("not_recording", "No recording in progress", nil)
      return
    }

    recorder.stop()
    let audioSession = AVAudioSession.sharedInstance()
    do {
      try audioSession.setActive(false)
      if let audioFilename = self.audioFilename {
        // Check if file exists and has data
        if FileManager.default.fileExists(atPath: audioFilename.path) {
             let attributes = try FileManager.default.attributesOfItem(atPath: audioFilename.path)
             let fileSize = attributes[FileAttributeKey.size] as? NSNumber
             if (fileSize?.intValue ?? 0) > 0 {
                 sendEvent(withName: "onAudioData", body: ["filePath": audioFilename.absoluteString])
                 resolve(["filePath": audioFilename.absoluteString, "fileSize": fileSize?.intValue ?? 0])
             } else {
                 reject("file_error", "Recorded file is empty or invalid.", nil)
             }
        } else {
            reject("file_error", "Recorded file not found.", nil)
        }

      } else {
        reject("file_error", "Audio filename not found", nil)
      }
    } catch {
      reject("stop_recording_failed", "Failed to stop recording: \(error.localizedDescription)", error)
    }
    self.audioRecorder = nil
    self.audioFilename = nil
  }

  // RCTEventEmitter methods
  override func supportedEvents() -> [String]! {
    return ["onAudioData", "onError"]
  }

  // AVAudioRecorderDelegate methods
  func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
    if !flag {
      sendEvent(withName: "onError", body: ["message": "Recording finished unsuccessfully"])
    }
    // `stopRecording` handles sending the onAudioData event upon successful recording.
  }

  func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
    if let error = error {
      sendEvent(withName: "onError", body: ["message": "Recording encode error: \(error.localizedDescription)"])
    }
  }

  @objc
  override static func requiresMainQueueSetup() -> Bool {
    return true
  }

  // MARK: - STT Methods

  @objc
  func initSTT(_ modelPath: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    print("[AudioInputModule] initSTT called with modelPath: \(modelPath)")
    // Assuming language "en" for now, this should ideally be a parameter
    if let modelPathCStr = modelPath.cString(using: .utf8),
       let langCStr = "en".cString(using: .utf8) {
      if self.sttContext != nil {
        RN_STT_free(self.sttContext)
        self.sttContext = nil
      }
      self.sttContext = RN_STT_init(modelPathCStr, langCStr)
      if self.sttContext != nil {
        resolve("STT initialized successfully")
      } else {
        reject("stt_init_failed", "Failed to initialize STT model (RN_STT_init returned null)", nil)
      }
    } else {
      reject("stt_init_failed", "Failed to convert modelPath or language to C string", nil)
    }
  }

  @objc
  func setUserVocabulary(_ vocabulary: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let context = self.sttContext else {
      reject("STT_ERROR", "STT not initialized. Call initSTT first.", nil)
      return
    }
    if let vocabularyCString = vocabulary.cString(using: .utf8) {
      RN_STT_setUserVocabulary(context, vocabularyCString)
      print("[AudioInputModule] User vocabulary set to: \(vocabulary)")
      resolve(nil)
    } else {
      reject("VOCAB_ERROR", "Failed to convert vocabulary to C string.", nil)
    }
  }

  @objc
  func processAudioFile(_ filePath: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    print("[AudioInputModule] processAudioFile called with filePath: \(filePath)")
    guard let context = self.sttContext else {
      reject("STT_ERROR", "STT not initialized. Call initSTT first.", nil)
      return
    }

    // Guard against file URLs if a direct path is expected by C++
    var pathForFFI = filePath
    if filePath.hasPrefix("file://") {
        if let url = URL(string: filePath) {
            pathForFFI = url.path
        }
    }

    if let filePathCString = pathForFFI.cString(using: .utf8) {
        if let cTranscription = RN_STT_processAudioFile(context, filePathCString) {
            let transcription = String(cString: cTranscription)
            RN_STT_free_string(UnsafeMutablePointer(mutating: cTranscription)) // Free the C string
            resolve(transcription)
        } else {
            reject("stt_process_failed", "Failed to process audio file or no transcription produced.", nil)
        }
    } else {
        reject("PATH_ERROR", "Failed to convert file path to C string.", nil)
    }
  }

  @objc
  func releaseSTT(_ resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    print("[AudioInputModule] releaseSTT called")
    if let context = self.sttContext {
      RN_STT_free(context)
      self.sttContext = nil
      resolve("STT released successfully")
    } else {
      resolve("STT already released or not initialized")
    }
  }

}

// Extend AudioInputModule to conform to AVAudioRecorderDelegate
extension AudioInputModule: AVAudioRecorderDelegate {}
