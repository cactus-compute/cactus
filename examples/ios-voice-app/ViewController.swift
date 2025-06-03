import UIKit
import AVFoundation // For AVAudioSession setup if needed, though STTService might handle it
import Cactus // Assuming your XCFramework and Swift wrappers are bundled under this module name

class ViewController: UIViewController {

    private var cactusSTTService: CactusSTTService?
    private var isRecording = false

    lazy var recordButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Start Recording", for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        button.addTarget(self, action: #selector(recordButtonTapped), for: .touchUpInside)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    lazy var transcriptionTextView: UITextView = {
        let textView = UITextView()
        textView.font = UIFont.systemFont(ofSize: 16)
        textView.isEditable = false
        textView.layer.borderColor = UIColor.lightGray.cgColor
        textView.layer.borderWidth = 1.0
        textView.layer.cornerRadius = 5
        textView.translatesAutoresizingMaskIntoConstraints = false
        return textView
    }()

    lazy var statusLabel: UILabel = {
        let label = UILabel()
        label.text = "Initialize STT and grant permissions."
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        title = "iOS STT Example"

        setupUI()
        initializeSTT()
    }

    private func setupUI() {
        view.addSubview(recordButton)
        view.addSubview(transcriptionTextView)
        view.addSubview(statusLabel)

        NSLayoutConstraint.activate([
            statusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            statusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

            recordButton.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 20),
            recordButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            recordButton.widthAnchor.constraint(equalToConstant: 200),
            recordButton.heightAnchor.constraint(equalToConstant: 50),

            transcriptionTextView.topAnchor.constraint(equalTo: recordButton.bottomAnchor, constant: 20),
            transcriptionTextView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            transcriptionTextView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            transcriptionTextView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])
    }

    private func initializeSTT() {
        cactusSTTService = CactusSTTService()

        // TODO: Replace with actual model path.
        // This model should be bundled with the app or downloaded to a known location.
        guard let modelPath = Bundle.main.path(forResource: "your_stt_model", ofType: "bin") else {
            print("STT Model not found in bundle.")
            statusLabel.text = "Error: STT Model not found. Please add it to the project and update the path."
            recordButton.isEnabled = false
            return
        }

        statusLabel.text = "Initializing STT..."
        cactusSTTService?.initSTT(modelPath: modelPath) { [weak self] error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if let error = error {
                    self.statusLabel.text = "STT Init Error: \(error.localizedDescription)"
                    self.recordButton.isEnabled = false
                    print("STT Initialization Error: \(error.localizedDescription)")
                } else {
                    self.statusLabel.text = "STT Initialized. Ready to record."
                    self.recordButton.isEnabled = true
                    print("STT Initialized Successfully")

                    // Optional: Call setUserVocabulary if needed
                    // self.cactusSTTService?.setUserVocabulary(vocabulary: ["custom word", "Cactus AI"], completion: { vocabError in
                    //     if let vocabError = vocabError {
                    //         print("Error setting vocab (placeholder): \(vocabError.localizedDescription)")
                    //     } else {
                    //         print("User vocabulary set (placeholder).")
                    //     }
                    // })
                }
            }
        }
    }

    @objc private func recordButtonTapped() {
        guard let sttService = cactusSTTService else {
            statusLabel.text = "STT Service not available."
            return
        }

        if isRecording {
            sttService.stopVoiceCapture()
            // UI update for stopping will be handled based on STT callbacks or state
            // For now, just update button and status directly
            recordButton.setTitle("Start Recording", for: .normal)
            recordButton.backgroundColor = .systemBlue
            statusLabel.text = "Stopping recording... Processing..."
            // isRecording will be set to false by the STTService delegate/completion
        } else {
            // Request microphone permission before starting
            AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    if granted {
                        self.statusLabel.text = "Recording..."
                        self.recordButton.setTitle("Stop Recording", for: .normal)
                        self.recordButton.backgroundColor = .systemRed
                        self.isRecording = true
                        self.transcriptionTextView.text = "" // Clear previous transcription

                        sttService.startVoiceCapture { [weak self] transcription, error in
                            DispatchQueue.main.async {
                                guard let self = self else { return }
                                self.isRecording = false // Reset recording state from service callback
                                self.recordButton.setTitle("Start Recording", for: .normal)
                                self.recordButton.backgroundColor = .systemBlue

                                if let error = error {
                                    self.statusLabel.text = "STT Error: \(error.localizedDescription)"
                                    self.transcriptionTextView.text = "Error: \(error.localizedDescription)"
                                    print("Transcription Error: \(error.localizedDescription)")
                                } else if let transcription = transcription {
                                    self.statusLabel.text = "Transcription received."
                                    self.transcriptionTextView.text = transcription
                                    print("Transcription: \(transcription)")
                                } else {
                                    self.statusLabel.text = "Transcription complete (no text or error)."
                                }
                            }
                        }
                    } else {
                        self.statusLabel.text = "Microphone permission denied."
                        print("Microphone permission denied.")
                    }
                }
            }
        }
    }

    deinit {
        cactusSTTService?.releaseSTT(completion: { error in
            if let error = error {
                print("Error releasing STT: \(error.localizedDescription)")
            } else {
                print("STT resources released.")
            }
        })
    }
}
