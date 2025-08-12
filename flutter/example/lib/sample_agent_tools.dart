import 'package:cactus/tools.dart';

class WeatherTool extends ToolExecutor {
  @override
  Future<dynamic> execute(Map<String, dynamic> args) async {
    final location = args['location'] as String? ?? 'unknown';
    return 'The weather in $location is sunny, 72°F';
  }
}
