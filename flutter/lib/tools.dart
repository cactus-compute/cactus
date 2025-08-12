import 'dart:convert';

class Parameter {
  final String type;
  final String description;
  final bool required;

  Parameter({
    required this.type,
    required this.description,
    this.required = false,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'description': description,
  };
}

abstract class ToolExecutor {
  Future<dynamic> execute(Map<String, dynamic> args);
}

class ToolSchema {
  final String type;
  final FunctionSchema function;

  ToolSchema({required this.type, required this.function});

  Map<String, dynamic> toJson() => {
    'type': type,
    'function': function.toJson(),
  };
}

class FunctionSchema {
  final String name;
  final String description;
  final ParametersSchema parameters;

  FunctionSchema({
    required this.name,
    required this.description,
    required this.parameters,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'description': description,
    'parameters': parameters.toJson(),
  };
}

class ParametersSchema {
  final String type;
  final Map<String, Parameter> properties;
  final List<String> required;

  ParametersSchema({
    required this.type,
    required this.properties,
    required this.required,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'properties': properties.map((k, v) => MapEntry(k, v.toJson())),
    'required': required,
  };
}

class Tool {
  final ToolExecutor func;
  final String description;
  final Map<String, Parameter> parameters;
  final List<String> required;

  Tool({
    required this.func,
    required this.description,
    required this.parameters,
    required this.required,
  });
}

class Tools {
  final Map<String, Tool> _tools = {};

  void add(String name, ToolExecutor func, String description, Map<String, Parameter> parameters) {
    final required = parameters.entries
        .where((entry) => entry.value.required)
        .map((entry) => entry.key)
        .toList();

    _tools[name] = Tool(
      func: func,
      description: description,
      parameters: parameters,
      required: required,
    );
  }

  List<ToolSchema> getSchemas() {
    return _tools.entries.map((entry) => ToolSchema(
      type: 'function',
      function: FunctionSchema(
        name: entry.key,
        description: entry.value.description,
        parameters: ParametersSchema(
          type: 'object',
          properties: entry.value.parameters,
          required: entry.value.required,
        ),
      ),
    )).toList();
  }

  Future<dynamic> execute(String name, Map<String, dynamic> args) async {
    final tool = _tools[name];
    if (tool == null) throw ArgumentError('Tool $name not found');
    return await tool.func.execute(args);
  }

  bool isEmpty() => _tools.isEmpty;
}

class ToolCallResult {
  final bool toolCalled;
  final String? toolName;
  final Map<String, String>? toolInput;
  final String? toolOutput;

  ToolCallResult({
    required this.toolCalled,
    this.toolName,
    this.toolInput,
    this.toolOutput,
  });
}

Future<ToolCallResult> parseAndExecuteTool(String? modelResponse, Tools tools) async {
  if (modelResponse == null || modelResponse.trim().isEmpty) {
    return ToolCallResult(toolCalled: false);
  }

  try {
    final jsonBlocks = _extractJsonBlocks(modelResponse);

    for (final jsonBlock in jsonBlocks) {
      try {
        final response = jsonDecode(jsonBlock) as Map<String, dynamic>;
        final toolCalls = response['tool_calls'] ?? response['tool_call'];

        if (toolCalls != null && toolCalls is List && toolCalls.isNotEmpty) {
          final toolCall = toolCalls.first as Map<String, dynamic>;
          final toolName = toolCall['name'] as String;
          final arguments = toolCall['arguments'] as Map<String, dynamic>;
          
          final toolOutput = await tools.execute(toolName, arguments);

          return ToolCallResult(
            toolCalled: true,
            toolName: toolName,
            toolInput: arguments.map((k, v) => MapEntry(k, v.toString())),
            toolOutput: toolOutput.toString(),
          );
        }
      } catch (e) {
        continue;
      }
    }

    return ToolCallResult(toolCalled: false);
  } catch (e) {
    return ToolCallResult(toolCalled: false);
  }
}

List<String> _extractJsonBlocks(String response) {
  final jsonBlocks = <String>[];

  if (response.contains('"tool_calls"') || response.contains('"tool_call"')) {
    int braceCount = 0;
    int startIndex = -1;

    for (int i = 0; i < response.length; i++) {
      switch (response[i]) {
        case '{':
          if (braceCount == 0) startIndex = i;
          braceCount++;
          break;
        case '}':
          braceCount--;
          if (braceCount == 0 && startIndex != -1) {
            final candidate = response.substring(startIndex, i + 1);
            if ((candidate.contains('"tool_calls"') || candidate.contains('"tool_call"'))
                && !jsonBlocks.contains(candidate)) {
              jsonBlocks.add(candidate);
            }
            startIndex = -1;
          }
          break;
      }
    }
  }

  return jsonBlocks;
}
