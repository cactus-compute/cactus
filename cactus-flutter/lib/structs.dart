import 'dart:math';

class LoraAdapterInfo {
  final String path;
  final double scale;

  LoraAdapterInfo({required this.path, required this.scale});

  @override
  String toString() => 'LoraAdapterInfo(path: $path, scale: $scale)';
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

  factory BenchResult.fromJson(List<dynamic> json) {
    if (json.length < 7) {
      throw const FormatException("Invalid JSON array length for BenchResult");
    }
    return BenchResult(
      modelDesc: json[0] as String,
      modelSize: (json[1] as num).toInt(),
      modelNParams: (json[2] as num).toInt(),
      ppAvg: (json[3] as num).toDouble(),
      ppStd: (json[4] as num).toDouble(),
      tgAvg: (json[5] as num).toDouble(),
      tgStd: (json[6] as num).toDouble(),
    );
  }

  @override
  String toString() {
    return 'BenchResult(modelDesc: $modelDesc, modelSize: $modelSize, modelNParams: $modelNParams, ppAvg: $ppAvg, ppStd: $ppStd, tgAvg: $tgAvg, tgStd: $tgStd)';
  }
}

class FormattedChatResult {
  final String prompt;
  final String? grammar;

  FormattedChatResult({required this.prompt, this.grammar});

  @override
  String toString() => 'FormattedChatResult(prompt: ${prompt.substring(0, min(50, prompt.length))}..., grammar: $grammar)';
}

class ChatResult {
  final String prompt;
  final String? jsonSchema;
  final String? tools;
  final String? toolChoice;
  final bool parallelToolCalls;

  ChatResult({
    required this.prompt,
    this.jsonSchema,
    this.tools,
    this.toolChoice,
    this.parallelToolCalls = false,
  });

  @override
  String toString() => 'ChatResult(prompt: ${prompt.substring(0, min(50, prompt.length))}..., '
      'jsonSchema: $jsonSchema, tools: $tools, toolChoice: $toolChoice, '
      'parallelToolCalls: $parallelToolCalls)';
}

/// Advanced chat formatting result with rich metadata (similar to React Native's JinjaFormattedChatResult)
class AdvancedChatResult {
  final String prompt;
  final int? chatFormat;
  final String? grammar;
  final bool? grammarLazy;
  final List<GrammarTrigger>? grammarTriggers;
  final List<String>? preservedTokens;
  final List<String>? additionalStops;

  AdvancedChatResult({
    required this.prompt,
    this.chatFormat,
    this.grammar,
    this.grammarLazy,
    this.grammarTriggers,
    this.preservedTokens,
    this.additionalStops,
  });

  @override
  String toString() => 'AdvancedChatResult(prompt: ${prompt.substring(0, min(50, prompt.length))}..., '
      'grammar: ${grammar != null ? "present" : "null"}, '
      'additionalStops: ${additionalStops?.length ?? 0})';
}

/// Grammar trigger for lazy grammar evaluation
class GrammarTrigger {
  final int type;
  final String value;
  final int token;

  GrammarTrigger({
    required this.type,
    required this.value,
    required this.token,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'value': value,
    'token': token,
  };

  factory GrammarTrigger.fromJson(Map<String, dynamic> json) => GrammarTrigger(
    type: json['type'],
    value: json['value'],
    token: json['token'],
  );
}

/// Response format for structured output (similar to React Native's CompletionResponseFormat)
enum ResponseFormatType {
  text,
  jsonObject,
  jsonSchema,
}

class ResponseFormat {
  final ResponseFormatType type;
  final Map<String, dynamic>? jsonSchema;
  final bool? strict;
  final Map<String, dynamic>? schema; // For jsonObject type

  ResponseFormat({
    required this.type,
    this.jsonSchema,
    this.strict,
    this.schema,
  });

  ResponseFormat.text() : this(type: ResponseFormatType.text);
  
  ResponseFormat.jsonObject({Map<String, dynamic>? schema}) 
    : this(type: ResponseFormatType.jsonObject, schema: schema);
  
  ResponseFormat.jsonSchema({
    required Map<String, dynamic> schema,
    bool? strict,
  }) : this(
    type: ResponseFormatType.jsonSchema, 
    jsonSchema: {'schema': schema, 'strict': strict ?? false}
  );

  Map<String, dynamic>? getJsonSchema() {
    switch (type) {
      case ResponseFormatType.jsonSchema:
        return jsonSchema?['schema'];
      case ResponseFormatType.jsonObject:
        return schema ?? {};
      case ResponseFormatType.text:
        return null;
    }
  }
}

/// Tool definition for function calling
class ToolDefinition {
  final String type;
  final FunctionDefinition function;

  ToolDefinition({
    this.type = 'function',
    required this.function,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'function': function.toJson(),
  };
}

class FunctionDefinition {
  final String name;
  final String description;
  final Map<String, dynamic> parameters;

  FunctionDefinition({
    required this.name,
    required this.description,
    required this.parameters,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'description': description,
    'parameters': parameters,
  };
}

/// Tool call result from parsing completion output
class ToolCall {
  final String id;
  final String type;
  final String name;
  final Map<String, dynamic> arguments;

  ToolCall({
    required this.id,
    this.type = 'function',
    required this.name,
    required this.arguments,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'type': type,
    'function': {
      'name': name,
      'arguments': arguments,
    },
  };
}

/// Tool execution result
class ToolExecutionResult {
  final bool toolCalled;
  final String? toolName;
  final Map<String, dynamic>? toolInput;
  final dynamic toolOutput;
  final String? error;

  ToolExecutionResult({
    required this.toolCalled,
    this.toolName,
    this.toolInput,
    this.toolOutput,
    this.error,
  });

  ToolExecutionResult.notCalled() : this(toolCalled: false);
  
  ToolExecutionResult.called({
    required String name,
    required Map<String, dynamic> input,
    required dynamic output,
  }) : this(
    toolCalled: true,
    toolName: name,
    toolInput: input,
    toolOutput: output,
  );

  ToolExecutionResult.error({
    required String name,
    required Map<String, dynamic> input,
    required String error,
  }) : this(
    toolCalled: true,
    toolName: name,
    toolInput: input,
    error: error,
  );
}

class ModelCapabilities {
  final bool llamaChatSupported;
  final bool jinjaSupported;
  final bool defaultJinjaSupported;
  final bool toolUseJinjaSupported;
  final Map<String, bool> jinjaCaps;

  ModelCapabilities({
    required this.llamaChatSupported,
    required this.jinjaSupported,
    required this.defaultJinjaSupported,
    required this.toolUseJinjaSupported,
    required this.jinjaCaps,
  });

  @override
  String toString() => 'ModelCapabilities(llamaChat: $llamaChatSupported, '
      'jinja: $jinjaSupported, defaultJinja: $defaultJinjaSupported, '
      'toolUseJinja: $toolUseJinjaSupported, caps: $jinjaCaps)';
}

class ModelMeta {
  final String description;
  final int size;
  final int nParams;

  ModelMeta({required this.description, required this.size, required this.nParams});

  @override
  String toString() => 'ModelMeta(description: $description, size: $size, nParams: $nParams)';
}