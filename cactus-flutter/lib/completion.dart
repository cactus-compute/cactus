import './chat.dart';
import './structs.dart';

typedef CactusTokenCallback = bool Function(String token);

class CactusCompletionParams {
  final List<ChatMessage> messages;
  final String? chatTemplate;
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
  final double? dryMultiplier;
  final double? dryBase;
  final int? dryAllowedLength;
  final int? dryPenaltyLastN;
  final List<String>? drySequenceBreakers;
  final bool? ignoreEos;
  final int? nProbs;
  final List<String>? stopSequences;
  final String? grammar;
  final Function(String)? onNewToken;
  final bool? jinja;
  final String? tools;
  final bool? parallelToolCalls;
  final String? toolChoice;
  final ResponseFormat? responseFormat;

  CactusCompletionParams({
    required this.messages,
    this.chatTemplate,
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
    this.dryMultiplier,
    this.dryBase,
    this.dryAllowedLength,
    this.dryPenaltyLastN,
    this.drySequenceBreakers,
    this.ignoreEos,
    this.nProbs,
    this.stopSequences,
    this.grammar,
    this.onNewToken,
    this.jinja,
    this.tools,
    this.parallelToolCalls,
    this.toolChoice,
    this.responseFormat,
  });

  CactusCompletionParams copyWith({
    List<ChatMessage>? messages,
    String? chatTemplate,
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
    double? dryMultiplier,
    double? dryBase,
    int? dryAllowedLength,
    int? dryPenaltyLastN,
    List<String>? drySequenceBreakers,
    bool? ignoreEos,
    int? nProbs,
    List<String>? stopSequences,
    String? grammar,
    Function(String)? onNewToken,
    bool? jinja,
    String? tools,
    bool? parallelToolCalls,
    String? toolChoice,
    ResponseFormat? responseFormat,
  }) {
    return CactusCompletionParams(
      messages: messages ?? this.messages,
      chatTemplate: chatTemplate ?? this.chatTemplate,
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
      dryMultiplier: dryMultiplier ?? this.dryMultiplier,
      dryBase: dryBase ?? this.dryBase,
      dryAllowedLength: dryAllowedLength ?? this.dryAllowedLength,
      dryPenaltyLastN: dryPenaltyLastN ?? this.dryPenaltyLastN,
      drySequenceBreakers: drySequenceBreakers ?? this.drySequenceBreakers,
      ignoreEos: ignoreEos ?? this.ignoreEos,
      nProbs: nProbs ?? this.nProbs,
      stopSequences: stopSequences ?? this.stopSequences,
      grammar: grammar ?? this.grammar,
      onNewToken: onNewToken ?? this.onNewToken,
      jinja: jinja ?? this.jinja,
      tools: tools ?? this.tools,
      parallelToolCalls: parallelToolCalls ?? this.parallelToolCalls,
      toolChoice: toolChoice ?? this.toolChoice,
      responseFormat: responseFormat ?? this.responseFormat,
    );
  }

  @override
  String toString() {
    return 'CactusCompletionParams(messages: ${messages.length}, '
        'maxTokens: $maxPredictedTokens, temp: $temperature)';
  }
}

class CactusCompletionResult {
  final String text;
  final int tokensPredicted;
  final int tokensEvaluated;
  final bool truncated;
  final bool stoppedEos;
  final bool stoppedWord;
  final bool stoppedLimit;
  final String stoppingWord;

  CactusCompletionResult({
    required this.text,
    required this.tokensPredicted,
    required this.tokensEvaluated,
    required this.truncated,
    required this.stoppedEos,
    required this.stoppedWord,
    required this.stoppedLimit,
    required this.stoppingWord,
  });

  @override
  String toString() {
    return 'CactusCompletionResult(text: ${text.length > 50 ? "${text.substring(0, 50)}..." : text}, '
        'tokensPredicted: $tokensPredicted, tokensEvaluated: $tokensEvaluated, '
        'stoppedEos: $stoppedEos, stoppedWord: $stoppedWord, stoppedLimit: $stoppedLimit)';
  }
}