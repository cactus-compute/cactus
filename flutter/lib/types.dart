typedef CactusTokenCallback = bool Function(String token);
typedef CactusProgressCallback = void Function(double? progress, String statusMessage, bool isError);

class ChatMessage {
  final String role;
  final String content;
  final double? tokensPerSecond;

  ChatMessage({required this.role, required this.content, this.tokensPerSecond});
  @override
  bool operator ==(Object other) => other is ChatMessage && role == other.role && content == other.content;
  
  @override
  int get hashCode => role.hashCode ^ content.hashCode;

  Map<String, String> toJson() => {
    'role': role,
    'content': content,
  };
}

class CactusException implements Exception {
  final String message;
  final dynamic underlyingError;

  CactusException(this.message, [this.underlyingError]);

  @override
  String toString() {
    if (underlyingError != null) {
      return 'CactusException: $message (Caused by: $underlyingError)';
    }
    return 'CactusException: $message';
  }
}

class CactusResult {
  final String text;
  final int tokensPredicted;
  final int tokensEvaluated;
  final bool truncated;
  final bool stoppedEos;
  final bool stoppedWord;
  final bool stoppedLimit;
  final String stoppingWord;

  CactusResult({
    required this.text,
    required this.tokensPredicted,
    required this.tokensEvaluated,
    required this.truncated,
    required this.stoppedEos,
    required this.stoppedWord,
    required this.stoppedLimit,
    required this.stoppingWord,
  });
}

class LoraAdapterInfo {
  final String path;
  final double scale;

  LoraAdapterInfo({
    required this.path,
    required this.scale,
  });
}

// Legacy types for backward compatibility
typedef CactusCompletionResult = CactusResult;

class CactusInitParams {
  final String? modelPath;
  final String? modelUrl;
  final String? modelFilename;
  final String? mmprojPath;
  final String? mmprojUrl;
  final String? mmprojFilename;
  final String? chatTemplate;
  final int contextSize;
  final int batchSize;
  final int ubatchSize;
  final int gpuLayers;
  final int threads;
  final bool useMmap;
  final bool useMlock;
  final bool generateEmbeddings;
  final int poolingType;
  final int normalizeEmbeddings;
  final bool useFlashAttention;
  final String? cacheTypeK;
  final String? cacheTypeV;
  final CactusProgressCallback? onInitProgress;

  CactusInitParams({
    this.modelPath,
    this.modelUrl,
    this.modelFilename,
    this.mmprojPath,
    this.mmprojUrl,
    this.mmprojFilename,
    this.chatTemplate,
    this.contextSize = 2048,
    this.batchSize = 512,
    this.ubatchSize = 512,
    this.gpuLayers = 0,
    this.threads = 4,
    this.useMmap = true,
    this.useMlock = false,
    this.generateEmbeddings = false,
    this.poolingType = 0,
    this.normalizeEmbeddings = 2,
    this.useFlashAttention = false,
    this.cacheTypeK,
    this.cacheTypeV,
    this.onInitProgress,
  });
}

class ResponseFormat {
  final Map<String, dynamic>? schema;
  final String? type;

  ResponseFormat({this.schema, this.type});
}

class CactusCompletionParams {
  final List<ChatMessage> messages;
  final int maxPredictedTokens;
  final int? threads;
  final int? seed;
  final double? temperature;
  final int? topK;
  final double? topP;
  final double? minP;
  final double? typicalP;
  final int? penaltyLastN;
  final double? penaltyRepeat;
  final double? penaltyFreq;
  final double? penaltyPresent;
  final int? mirostat;
  final double? mirostatTau;
  final double? mirostatEta;
  final bool? ignoreEos;
  final int? nProbs;
  final List<String>? stopSequences;
  final String? grammar;
  final String? chatTemplate;
  final CactusTokenCallback? onNewToken;
  final ResponseFormat? responseFormat;
  final bool? jinja;

  const CactusCompletionParams({
    required this.messages,
    this.maxPredictedTokens = 256,
    this.threads,
    this.seed,
    this.temperature,
    this.topK,
    this.topP,
    this.minP,
    this.typicalP,
    this.penaltyLastN,
    this.penaltyRepeat,
    this.penaltyFreq,
    this.penaltyPresent,
    this.mirostat,
    this.mirostatTau,
    this.mirostatEta,
    this.ignoreEos,
    this.nProbs,
    this.stopSequences,
    this.grammar,
    this.chatTemplate,
    this.onNewToken,
    this.responseFormat,
    this.jinja,
  });

  CactusCompletionParams copyWith({
    List<ChatMessage>? messages,
    int? maxPredictedTokens,
    int? threads,
    int? seed,
    double? temperature,
    int? topK,
    double? topP,
    double? minP,
    double? typicalP,
    int? penaltyLastN,
    double? penaltyRepeat,
    double? penaltyFreq,
    double? penaltyPresent,
    int? mirostat,
    double? mirostatTau,
    double? mirostatEta,
    bool? ignoreEos,
    int? nProbs,
    List<String>? stopSequences,
    String? grammar,
    String? chatTemplate,
    CactusTokenCallback? onNewToken,
    ResponseFormat? responseFormat,
    bool? jinja,
  }) {
    return CactusCompletionParams(
      messages: messages ?? this.messages,
      maxPredictedTokens: maxPredictedTokens ?? this.maxPredictedTokens,
      threads: threads ?? this.threads,
      seed: seed ?? this.seed,
      temperature: temperature ?? this.temperature,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      minP: minP ?? this.minP,
      typicalP: typicalP ?? this.typicalP,
      penaltyLastN: penaltyLastN ?? this.penaltyLastN,
      penaltyRepeat: penaltyRepeat ?? this.penaltyRepeat,
      penaltyFreq: penaltyFreq ?? this.penaltyFreq,
      penaltyPresent: penaltyPresent ?? this.penaltyPresent,
      mirostat: mirostat ?? this.mirostat,
      mirostatTau: mirostatTau ?? this.mirostatTau,
      mirostatEta: mirostatEta ?? this.mirostatEta,
      ignoreEos: ignoreEos ?? this.ignoreEos,
      nProbs: nProbs ?? this.nProbs,
      stopSequences: stopSequences ?? this.stopSequences,
      grammar: grammar ?? this.grammar,
      chatTemplate: chatTemplate ?? this.chatTemplate,
      onNewToken: onNewToken ?? this.onNewToken,
      responseFormat: responseFormat ?? this.responseFormat,
      jinja: jinja ?? this.jinja,
    );
  }
}

class BenchResult {
  final String modelDesc;
  final int modelSize;
  final int modelNParams;
  final double ppAvg;
  final double ppStd;
  final double tgAvg;
  final double tgStd;

  BenchResult({
    required this.modelDesc,
    required this.modelSize,
    required this.modelNParams,
    required this.ppAvg,
    required this.ppStd,
    required this.tgAvg,
    required this.tgStd,
  });
} 