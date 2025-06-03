// AudioInputModule.swift
import Foundation
import AVFoundation

@objc(AudioInputModule)
class AudioInputModule: RCTEventEmitter {

  private var audioRecorder: AVAudioRecorder?
  private var audioFilename: URL?

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
    // TODO: Call actual C++ FFI function for STT initialization
    // e.g., let success = cactus_stt_init_ffi(modelPath)
    // For now, simulate success
    print("[AudioInputModule] initSTT called with modelPath: \(modelPath)")
    // Assuming cactus_stt_init_ffi would be a C function linked into the project:
    // let result = cactus_stt_init_ffi((modelPath as NSString).utf8String)
    // if result == 0 {
    //   resolve("STT initialized successfully")
    // } else {
    //   reject("stt_init_failed", "Failed to initialize STT model", nil)
    // }
    resolve("STT initialized successfully (placeholder)")
  }

  @objc
  func processAudioFile(_ filePath: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // TODO: Call actual C++ FFI function for STT processing
    // e.g., let transcription = cactus_stt_process_file_ffi(filePath)
    // For now, simulate success with placeholder transcription
    print("[AudioInputModule] processAudioFile called with filePath: \(filePath)")

    // Guard against file URLs if a direct path is expected by C++
    var pathForFFI = filePath
    if filePath.hasPrefix("file://") {
        if let url = URL(string: filePath) {
            pathForFFI = url.path
        }
    }

    // Assuming cactus_stt_process_file_ffi would be a C function:
    // let cTranscription = cactus_stt_process_file_ffi((pathForFFI as NSString).utf8String)
    // if cTranscription != nil {
    //   let transcription = String(cString: cTranscription!)
    //   cactus_stt_free_string_ffi(cTranscription) // Assuming memory management function
    //   resolve(transcription)
    // } else {
    //   reject("stt_process_failed", "Failed to process audio file", nil)
    // }
    resolve("Placeholder transcription for \(pathForFFI)")
  }

  @objc
  func releaseSTT(_ resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    // TODO: Call actual C++ FFI function for STT release
    // e.g., cactus_stt_release_ffi()
    print("[AudioInputModule] releaseSTT called")
    // cactus_stt_release_ffi()
    resolve("STT released successfully (placeholder)")
  }

}

// Extend AudioInputModule to conform to AVAudioRecorderDelegate
extension AudioInputModule: AVAudioRecorderDelegate {}
