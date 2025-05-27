import './chat.dart';

/// Callback for receiving new tokens during streaming completion.
///
/// Return `false` from the callback to stop the generation early.
/// [token] is the newly generated token string.
typedef CactusTokenCallback = bool Function(String token);

/// Parameters for a text or chat completion request.
class CactusCompletionParams {
  /// A list of [ChatMessage] objects representing the conversation history.
  /// Required for chat-style completions. For text-only completion, this can be empty
  /// or contain a single user message with the prompt.
  final List<ChatMessage> messages;

  /// Optional path to an image file for multimodal models (e.g., LLaVA).
  /// If provided, the model will process this image in conjunction with the text prompt.
  final String? imagePath;

  /// The maximum number of tokens to predict (often `n_predict`).
  /// A value of -1 means predict indefinitely (until EOS or other stopping conditions).
  /// Defaults to -1.
  final int maxPredictedTokens;

  /// Number of threads to use for this specific completion.
  /// If 0, the number of threads from [CactusInitParams.threads] is used.
  /// Defaults to 0.
  final int threads;

  /// The random seed for generation. A value of -1 means use a random seed determined by the native layer.
  /// Consistent seeding helps in reproducing results.
  /// Defaults to -1.
  final int seed;

  /// Controls randomness in sampling (often `temperature`). Lower values (e.g., 0.2) make the model more deterministic
  /// and focused, while higher values (e.g., 0.8-1.0) make it more creative and random.
  /// A value of 0 effectively means greedy decoding (always picking the most probable token).
  /// Defaults to 0.8.
  final double temperature;

  /// Limits sampling to the top K most probable tokens (often `top_k`).
  /// For example, if `topK` is 20, the model will only consider the 20 most likely
  /// next tokens. A value of 0 disables Top-K sampling.
  /// Defaults to 20.
  final int topK;

  /// Nucleus sampling parameter (often `top_p`). Limits sampling to tokens whose cumulative
  /// probability is >= `topP`.
  /// For example, `topP` = 0.95 means the model considers the smallest set of tokens whose
  /// cumulative probability exceeds 95%. 1.0 disables Top-P sampling.
  /// Defaults to 0.95.
  final double topP;

  /// Minimum probability for a token to be considered in sampling (often `min_p`).
  /// Tokens with probability below `minP` relative to the most probable token are excluded.
  /// Defaults to 0.05.
  final double minP;

  /// Typical P sampling parameter (often `typical_p`).
  /// Helps to reduce the likelihood of highly improbable tokens being sampled.
  /// A value of 1.0 disables typical sampling.
  /// Defaults to 1.0.
  final double typicalP;

  /// Number of recent tokens to consider for repeat penalty (often `penalty_last_n`).
  /// Defaults to 64.
  final int penaltyLastN;

  /// Penalty applied to repeated tokens (often `penalty_repeat`). Higher values (e.g., > 1.0)
  /// discourage repetition, lower values (< 1.0) encourage it.
  /// A value of 1.0 means no penalty.
  /// Defaults to 1.1.
  final double penaltyRepeat;

  /// Penalty applied based on token frequency in the context so far (often `penalty_freq`).
  /// Higher values reduce the likelihood of tokens that have already appeared frequently.
  /// Defaults to 0.0 (disabled).
  final double penaltyFreq;

  /// Penalty applied to tokens already present in the context (often `penalty_present`).
  /// Higher values reduce the likelihood of any token that has already appeared, regardless of frequency.
  /// Defaults to 0.0 (disabled).
  final double penaltyPresent;

  /// Mirostat sampling mode (0: disabled, 1: Mirostat, 2: Mirostat 2.0).
  /// Mirostat is an alternative sampling method that aims to maintain a target perplexity.
  /// Defaults to 0 (disabled).
  final int mirostat;

  /// Mirostat target entropy (often `mirostat_tau`). Used when `mirostat` is 1 or 2.
  /// Defaults to 5.0.
  final double mirostatTau;

  /// Mirostat learning rate (often `mirostat_eta`). Used when `mirostat` is 1 or 2.
  /// Defaults to 0.1.
  final double mirostatEta;

  /// Whether to ignore the End-Of-Sequence (EOS) token during generation.
  /// If true, the model will not stop when it generates EOS, potentially continuing until `maxPredictedTokens`.
  /// Defaults to false.
  final bool ignoreEos;

  /// Number of probabilities to return for the top N tokens (if supported by the model and native layer).
  /// If > 0, the `CactusCompletionResult` might contain probabilities for each generated token.
  /// Defaults to 0 (disabled).
  final int nProbs;

  /// A list of sequences that, when generated, will cause completion to stop.
  /// The stop sequences themselves are not included in the output.
  final List<String>? stopSequences;

  /// GBNF (GGML BNF) grammar to constrain the output to a specific format.
  /// Provide a string containing the grammar rules.
  /// If null, no grammar is applied.
  final String? grammar;

  /// Callback function that receives each new token as it is generated during streaming.
  /// Return `false` from the callback to stop the generation early.
  final CactusTokenCallback? onNewToken;

  /// Optional custom chat template string (e.g., a Jinja2-like format) to override
  /// the one set during context initialization for this specific completion request.
  /// If null or empty, the context's default or the model's internal template is used.
  final String? chatTemplate;

  /// Creates parameters for a completion request.
  CactusCompletionParams({
    required this.messages,
    this.imagePath,
    this.maxPredictedTokens = -1,
    this.threads = 0,
    this.seed = -1,
    this.temperature = 0.8,
    this.topK = 20,
    this.topP = 0.95,
    this.minP = 0.05,
    this.typicalP = 1.0,
    this.penaltyLastN = 64,
    this.penaltyRepeat = 1.1,
    this.penaltyFreq = 0.0,
    this.penaltyPresent = 0.0,
    this.mirostat = 0,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.ignoreEos = false,
    this.nProbs = 0,
    this.stopSequences,
    this.grammar,
    this.onNewToken,
    this.chatTemplate,
  });
}

/// Represents the result of a text or chat completion operation from [CactusContext.completion].
class CactusCompletionResult {
  /// The complete generated text.
  final String text;

  /// The number of tokens predicted during the generation phase.
  final int tokensPredicted;

  /// The number of prompt tokens evaluated before generation started.
  final int tokensEvaluated;

  /// True if the generation was truncated because the input prompt exceeded the model's context window.
  final bool truncated;

  /// True if generation stopped because the End-Of-Sequence (EOS) token was generated.
  final bool stoppedEos;

  /// True if generation stopped because one of the [CactusCompletionParams.stopSequences] was encountered.
  final bool stoppedWord;

  /// True if generation stopped due to reaching the prediction limit ([CactusCompletionParams.maxPredictedTokens]).
  final bool stoppedLimit;

  /// The specific word from [CactusCompletionParams.stopSequences] that caused generation to stop, if [stoppedWord] is true.
  /// Otherwise, this will be an empty string.
  final String stoppingWord;

  /// Total time spent in the native token generation phase for this completion, in microseconds.
  /// This can be used to calculate tokens per second: `tokensPredicted / (generationTimeUs / 1000000.0)`.
  /// A value of 0 might indicate that timing was not available or an error occurred.
  final int generationTimeUs;

  // TODO: Consider adding timing information if available from native layer (e.g., prompt eval time, token gen time)
  // final double promptEvalTimeSeconds;
  // final double tokenGenerationTimeSeconds;

  /// Creates a new [CactusCompletionResult].
  CactusCompletionResult({
    required this.text,
    required this.tokensPredicted,
    required this.tokensEvaluated,
    required this.truncated,
    required this.stoppedEos,
    required this.stoppedWord,
    required this.stoppedLimit,
    required this.stoppingWord,
    this.generationTimeUs = 0, // Default to 0 if not provided
  });

  /// Calculates the approximate tokens per second for the generation phase.
  /// Returns 0.0 if [generationTimeUs] is 0 or [tokensPredicted] is 0.
  double get tokensPerSecond {
    if (generationTimeUs <= 0 || tokensPredicted <= 0) {
      return 0.0;
    }
    return tokensPredicted / (generationTimeUs / 1000000.0);
  }

  @override
  String toString() {
    return 'CactusCompletionResult(text: ${text.length > 50 ? "${text.substring(0, 50)}..." : text}, tokensPredicted: $tokensPredicted, tokensEvaluated: $tokensEvaluated, stoppedEos: $stoppedEos, stoppedWord: $stoppedWord, stoppedLimit: $stoppedLimit, stoppingWord: $stoppingWord, truncated: $truncated, generationTimeUs: $generationTimeUs, tokensPerSecond: ${tokensPerSecond.toStringAsFixed(2)})';
  }
} 