import 'dart:convert';

/// Grammar generation for structured output

const String _spaceRule = '" "?';

String _buildRepetition(
  String itemRule,
  int minItems,
  int? maxItems, {
  String? separatorRule,
  bool itemRuleIsLiteral = false,
}) {
  separatorRule ??= '';

  if (separatorRule.isEmpty) {
    if (minItems == 0 && maxItems == 1) {
      return '$itemRule?';
    } else if (minItems == 1 && maxItems == null) {
      return '$itemRule+';
    }
  }

  String result = '';
  if (minItems > 0) {
    if (itemRuleIsLiteral && separatorRule.isEmpty) {
      final repeated = itemRule.substring(1, itemRule.length - 1) * minItems;
      result = '"$repeated"';
    } else {
      final parts = List.filled(minItems, itemRule);
      result = parts.join(separatorRule.isNotEmpty ? ' $separatorRule ' : ' ');
    }
  }

  String optRepetitions(int upToN, bool prefixWithSep) {
    final content = separatorRule!.isNotEmpty && prefixWithSep 
        ? '$separatorRule $itemRule' 
        : itemRule;
        
    if (upToN == 0) {
      return '';
    } else if (upToN == 1) {
      return '($content)?';
    } else if (separatorRule.isNotEmpty && !prefixWithSep) {
      return '($content ${optRepetitions(upToN - 1, true)})?';
    } else {
      final opening = List.filled(upToN, '($content').join(' ').trim();
      final closing = List.filled(upToN, ')?').join('');
      return opening + closing;
    }
  }

  if (minItems > 0 && maxItems != minItems) {
    result += ' ';
  }

  if (maxItems != null) {
    result += optRepetitions(maxItems - minItems, minItems > 0);
  } else {
    final itemOperator = '(${separatorRule.isNotEmpty ? '$separatorRule ' : ''}$itemRule)';
    
    if (minItems == 0 && separatorRule.isNotEmpty) {
      result = '($itemRule $itemOperator*)?';
    } else {
      result += '$itemOperator*';
    }
  }

  return result;
}

class GrammarBuiltinRule {
  final String content;
  final List<String> deps;

  GrammarBuiltinRule(this.content, this.deps);
}

const String _upTo15Digits = '[0-9]{0,15}';

final Map<String, GrammarBuiltinRule> _primitiveRules = {
  'boolean': GrammarBuiltinRule('("true" | "false") space', []),
  'decimal-part': GrammarBuiltinRule('[0-9] $_upTo15Digits', []),
  'integral-part': GrammarBuiltinRule('[0-9] | [1-9] $_upTo15Digits', []),
  'number': GrammarBuiltinRule(
    '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
    ['integral-part', 'decimal-part'],
  ),
  'integer': GrammarBuiltinRule('("-"? integral-part) space', ['integral-part']),
  'value': GrammarBuiltinRule('object | array | string | number | boolean | null', [
    'object', 'array', 'string', 'number', 'boolean', 'null'
  ]),
  'object': GrammarBuiltinRule(
    '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    ['string', 'value'],
  ),
  'array': GrammarBuiltinRule(
    '"[" space ( value ("," space value)* )? "]" space',
    ['value'],
  ),
  'uuid': GrammarBuiltinRule(
    '"\\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\\"" space',
    [],
  ),
  'char': GrammarBuiltinRule(
    '[^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
    [],
  ),
  'string': GrammarBuiltinRule('"\\"" char* "\\"" space', ['char']),
  'null': GrammarBuiltinRule('"null" space', []),
};

final Map<String, GrammarBuiltinRule> _stringFormatRules = {
  'date': GrammarBuiltinRule(
    '[0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
    [],
  ),
  'time': GrammarBuiltinRule(
    '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9] [0-9] [0-9] )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
    [],
  ),
  'date-time': GrammarBuiltinRule('date "T" time', ['date', 'time']),
  'date-string': GrammarBuiltinRule('"\\"" date "\\"" space', ['date']),
  'time-string': GrammarBuiltinRule('"\\"" time "\\"" space', ['time']),
  'date-time-string': GrammarBuiltinRule('"\\"" date-time "\\"" space', ['date-time']),
};

final RegExp _invalidRuleChars = RegExp(r'[^\dA-Za-z-]+');
final RegExp _grammarLiteralEscape = RegExp(r'[\n\r"]');

final Map<String, String> _grammarLiteralEscapes = {
  '\r': '\\r',
  '\n': '\\n',
  '"': '\\"',
  '-': '\\-',
  ']': '\\]',
};

String _formatLiteral(String literal) {
  final escaped = literal.replaceAllMapped(_grammarLiteralEscape, (m) => _grammarLiteralEscapes[m.group(0)] ?? '');
  return '"$escaped"';
}

String _generateConstantRule(dynamic value) => _formatLiteral(jsonEncode(value));

class SchemaGrammarConverter {
  final Map<String, int> _propOrder;
  final Map<String, String> _rules = {'space': _spaceRule};
  final Map<String, dynamic> _refs = {};

  SchemaGrammarConverter({
    Map<String, int>? propOrder,
    bool allowFetch = false,
    bool dotall = false,
  }) : _propOrder = propOrder ?? {};

  String _addRule(String name, String rule) {
    final escName = name.replaceAll(_invalidRuleChars, '-');
    String key = escName;

    if (_rules.containsKey(escName)) {
      if (_rules[escName] == rule) {
        return key;
      }

      int i = 0;
      while (_rules.containsKey('$escName$i') && _rules['$escName$i'] != rule) {
        i++;
      }
      key = '$escName$i';
    }

    _rules[key] = rule;
    return key;
  }

  String _generateUnionRule(String? name, List<dynamic> altSchemas) {
    return altSchemas.asMap().entries.map((entry) {
      final i = entry.key;
      final altSchema = entry.value;
      return visit(altSchema, '${name ?? ''}${name != null ? '-' : 'alternative-'}$i');
    }).join(' | ');
  }

  String visit(dynamic schema, String name) {
    final schemaType = schema['type'];
    final schemaFormat = schema['format'];
    
    if (schema.containsKey('\$ref')) {
      return _resolveRef(schema['\$ref']);
    }

    if (schema.containsKey('const')) {
      return _addRule(name, _generateConstantRule(schema['const']));
    }

    if (schema.containsKey('enum')) {
      final enumValues = schema['enum'] as List;
      final rules = enumValues.map((value) => _generateConstantRule(value)).toList();
      return _addRule(name, rules.join(' | '));
    }

    if (schema.containsKey('anyOf')) {
      return _addRule(name, _generateUnionRule(name, schema['anyOf']));
    }

    if (schema.containsKey('oneOf')) {
      return _addRule(name, _generateUnionRule(name, schema['oneOf']));
    }

    if (schemaType == 'object') {
      return _buildObjectRule(schema, name);
    }

    if (schemaType == 'array') {
      return _buildArrayRule(schema, name);
    }

    if (schemaType == 'string') {
      if (schemaFormat != null && _stringFormatRules.containsKey(schemaFormat)) {
        _addPrimitive(schemaFormat, _stringFormatRules[schemaFormat]);
        return schemaFormat;
      }
      
      if (schema.containsKey('pattern')) {
        return _visitPattern(schema['pattern'], name);
      }
      
      return _addPrimitive('string', _primitiveRules['string'])!;
    }

    if (schemaType == 'number') {
      return _addPrimitive('number', _primitiveRules['number'])!;
    }

    if (schemaType == 'integer') {
      return _addPrimitive('integer', _primitiveRules['integer'])!;
    }

    if (schemaType == 'boolean') {
      return _addPrimitive('boolean', _primitiveRules['boolean'])!;
    }

    if (schemaType == 'null') {
      return _addPrimitive('null', _primitiveRules['null'])!;
    }

    if (schemaType == null && schema.isEmpty) {
      return _addPrimitive('value', _primitiveRules['value'])!;
    }

    throw ArgumentError('Unsupported schema: $schema');
  }

  String? _addPrimitive(String name, GrammarBuiltinRule? rule) {
    if (rule == null) return null;
    
    for (final dep in rule.deps) {
      _addPrimitive(dep, _primitiveRules[dep] ?? _stringFormatRules[dep]);
    }
    
    _rules[name] = rule.content;
    return name;
  }

  String _buildObjectRule(Map<String, dynamic> schema, String name) {
    final properties = schema['properties'] as Map<String, dynamic>? ?? {};
    final required = Set<String>.from(schema['required'] ?? []);
    final additionalProperties = schema['additionalProperties'];

    if (properties.isEmpty) {
      if (additionalProperties == false) {
        return _addRule(name, '"{}" space');
      } else {
        return _addPrimitive('object', _primitiveRules['object'])!;
      }
    }

    final requiredProps = <String>[];
    final optionalProps = <String>[];

    for (final prop in properties.keys) {
      if (required.contains(prop)) {
        requiredProps.add(prop);
      } else {
        optionalProps.add(prop);
      }
    }

    // Sort by property order if specified
    requiredProps.sort((a, b) => (_propOrder[a] ?? 999).compareTo(_propOrder[b] ?? 999));
    optionalProps.sort((a, b) => (_propOrder[a] ?? 999).compareTo(_propOrder[b] ?? 999));

    final List<String> propRules = [];

    // Required properties
    for (int i = 0; i < requiredProps.length; i++) {
      final prop = requiredProps[i];
      final propSchema = properties[prop];
      final propRule = visit(propSchema, '$name-${prop.replaceAll(_invalidRuleChars, '-')}');
      
      final rule = '${_formatLiteral(prop)} ":" space $propRule';
      propRules.add(rule);
    }

    // Optional properties
    for (final prop in optionalProps) {
      final propSchema = properties[prop];
      final propRule = visit(propSchema, '$name-${prop.replaceAll(_invalidRuleChars, '-')}');
      
      final rule = '("," space ${_formatLiteral(prop)} ":" space $propRule)?';
      propRules.add(rule);
    }

    String objectRule;
    if (propRules.isEmpty) {
      objectRule = '"{}" space';
    } else {
      final content = propRules.join(' ');
      objectRule = '"{" space $content "}" space';
    }

    return _addRule(name, objectRule);
  }

  String _buildArrayRule(Map<String, dynamic> schema, String name) {
    final items = schema['items'];
    final minItems = schema['minItems'] ?? 0;
    final maxItems = schema['maxItems'];

    if (items == null) {
      return _addPrimitive('array', _primitiveRules['array'])!;
    }

    final itemRule = visit(items, '$name-item');
    final arrayContent = _buildRepetition(
      itemRule,
      minItems,
      maxItems,
      separatorRule: '"," space',
    );

    final rule = arrayContent.isEmpty 
        ? '"[]" space'
        : '"[" space $arrayContent "]" space';

    return _addRule(name, rule);
  }

  String _visitPattern(String pattern, String name) {
    // Simplified pattern handling - just treat as string for now
    // Full regex-to-grammar conversion would be very complex
    return _addPrimitive('string', _primitiveRules['string'])!;
  }

  String _resolveRef(String ref) {
    // Simplified ref resolution
    if (_refs.containsKey(ref)) {
      return _refs[ref]!;
    }
    throw ArgumentError('Unresolved reference: $ref');
  }

  String formatGrammar() {
    final lines = <String>[];
    
    // Add root rule first if it exists
    if (_rules.containsKey('root')) {
      lines.add('root ::= ${_rules['root']}');
    }
    
    // Add other rules
    for (final entry in _rules.entries) {
      if (entry.key != 'root') {
        lines.add('${entry.key} ::= ${entry.value}');
      }
    }
    
    return lines.join('\n');
  }
}

/// Convert JSON schema to GBNF grammar
String convertJsonSchemaToGrammar(Map<String, dynamic> schema, {
  Map<String, int>? propOrder,
  bool dotall = false,
  bool allowFetch = false,
}) {
  final converter = SchemaGrammarConverter(
    propOrder: propOrder,
    dotall: dotall,
    allowFetch: allowFetch,
  );

  converter.visit(schema, 'root');
  return converter.formatGrammar();
}

/// Generate basic JSON object grammar
String generateJsonObjectGrammar() {
  return '''root ::= object
object ::= "{" space (string ":" space value ("," space string ":" space value)*)? "}" space
string ::= "\\"" char* "\\"" space
value ::= object | array | string | number | boolean | null
array ::= "[" space (value ("," space value)*)? "]" space
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
integral-part ::= [0-9] | [1-9] [0-9]*
decimal-part ::= [0-9]+
boolean ::= ("true" | "false") space
null ::= "null" space
char ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
space ::= " "?''';
}