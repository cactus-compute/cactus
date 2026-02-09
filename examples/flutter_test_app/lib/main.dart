import 'dart:io';
import 'package:flutter/material.dart';
import 'cactus.dart';

void main() {
  runApp(const CactusTestApp());
}

class CactusTestApp extends StatelessWidget {
  const CactusTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Flutter Test',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2E7D32),
        useMaterial3: true,
      ),
      home: const CompletionTestPage(),
    );
  }
}

class CompletionTestPage extends StatefulWidget {
  const CompletionTestPage({super.key});

  @override
  State<CompletionTestPage> createState() => _CompletionTestPageState();
}

class _CompletionTestPageState extends State<CompletionTestPage> {
  final _promptController = TextEditingController(text: 'What is 2 + 2?');
  final _modelPathController = TextEditingController(
    text: '${Platform.environment['HOME']}/cactus/weights/lfm2-350m',
  );

  Cactus? _model;
  bool _loading = false;
  bool _generating = false;
  String _status = 'Ready — load a model to begin.';
  String _output = '';
  _Metrics? _metrics;

  bool get _isLoaded => _model != null;
  bool get _isBusy => _loading || _generating;

  Future<void> _loadModel() async {
    final path = _modelPathController.text.trim();
    if (path.isEmpty) return _setStatus('Enter a model path.');
    if (!Directory(path).existsSync() && !File(path).existsSync()) {
      return _setStatus('Path not found: $path');
    }

    setState(() { _loading = true; _status = 'Loading model...'; _output = ''; _metrics = null; });

    try {
      _model?.dispose();
      _model = Cactus.create(path);
      _setStatus('Model loaded — ${path.split('/').last}');
    } catch (e) {
      _setStatus('Load failed: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _runCompletion() async {
    if (!_isLoaded) return;
    final prompt = _promptController.text.trim();
    if (prompt.isEmpty) return _setStatus('Enter a prompt.');

    setState(() { _generating = true; _status = 'Generating...'; _output = ''; _metrics = null; });

    try {
      final result = _model!.complete(prompt);
      setState(() {
        _output = result.text;
        _metrics = _Metrics(
          promptTokens: result.promptTokens,
          completionTokens: result.completionTokens,
          prefillSpeed: result.prefillTokensPerSecond,
          decodeSpeed: result.decodeTokensPerSecond,
          ttft: result.timeToFirstToken,
          totalTime: result.totalTime,
        );
        _status = 'Done — ${result.completionTokens} tokens in '
            '${result.totalTime.toStringAsFixed(2)}s '
            '(${result.decodeTokensPerSecond.toStringAsFixed(1)} tok/s)';
      });
    } catch (e) {
      _setStatus('Completion failed: $e');
    } finally {
      setState(() => _generating = false);
    }
  }

  void _unloadModel() {
    _model?.dispose();
    setState(() {
      _model = null;
      _status = 'Model unloaded.';
      _output = '';
      _metrics = null;
    });
  }

  void _setStatus(String msg) => setState(() => _status = msg);

  @override
  void dispose() {
    _model?.dispose();
    _promptController.dispose();
    _modelPathController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cactus Flutter Test'),
        actions: [
          if (_isLoaded)
            TextButton.icon(
              onPressed: _isBusy ? null : _unloadModel,
              icon: const Icon(Icons.eject, size: 18),
              label: const Text('Unload'),
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _StatusBar(message: _status, isBusy: _isBusy, isLoaded: _isLoaded),
            const SizedBox(height: 16),
            Row(children: [
              Expanded(
                child: TextField(
                  controller: _modelPathController,
                  enabled: !_isBusy,
                  style: const TextStyle(fontSize: 13, fontFamily: 'monospace'),
                  decoration: const InputDecoration(
                    labelText: 'Model path',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: _isBusy ? null : _loadModel,
                child: Text(_isLoaded ? 'Reload' : 'Load'),
              ),
            ]),
            const SizedBox(height: 16),
            TextField(
              controller: _promptController,
              enabled: !_isBusy,
              maxLines: 3,
              decoration: const InputDecoration(
                labelText: 'Prompt',
                border: OutlineInputBorder(),
                alignLabelWithHint: true,
              ),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.icon(
                onPressed: _isLoaded && !_isBusy ? _runCompletion : null,
                icon: const Icon(Icons.play_arrow, size: 18),
                label: const Text('Run'),
              ),
            ),
            const SizedBox(height: 16),
            if (_metrics != null) _MetricsRow(metrics: _metrics!),
            if (_metrics != null) const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: theme.colorScheme.outlineVariant),
                ),
                child: SingleChildScrollView(
                  child: SelectableText(
                    _output.isEmpty ? 'Output will appear here.' : _output,
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 13,
                      height: 1.5,
                      color: _output.isEmpty
                          ? theme.colorScheme.onSurfaceVariant
                          : theme.colorScheme.onSurface,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  final String message;
  final bool isBusy;
  final bool isLoaded;

  const _StatusBar({required this.message, required this.isBusy, required this.isLoaded});

  Color get _color {
    if (isBusy) return Colors.orange;
    if (message.contains('failed') || message.startsWith('Path not found')) return Colors.red;
    if (isLoaded) return Colors.green;
    return Colors.grey;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: _color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _color.withOpacity(0.3)),
      ),
      child: Row(children: [
        if (isBusy)
          const Padding(
            padding: EdgeInsets.only(right: 8),
            child: SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
          ),
        Expanded(child: Text(message, style: TextStyle(color: _color))),
      ]),
    );
  }
}

class _MetricsRow extends StatelessWidget {
  final _Metrics metrics;
  const _MetricsRow({required this.metrics});

  @override
  Widget build(BuildContext context) {
    return Wrap(spacing: 16, runSpacing: 4, children: [
      _chip('Prompt', '${metrics.promptTokens} tok'),
      _chip('Completion', '${metrics.completionTokens} tok'),
      _chip('Prefill', '${metrics.prefillSpeed.toStringAsFixed(1)} tok/s'),
      _chip('Decode', '${metrics.decodeSpeed.toStringAsFixed(1)} tok/s'),
      _chip('TTFT', '${(metrics.ttft * 1000).toStringAsFixed(0)} ms'),
      _chip('Total', '${metrics.totalTime.toStringAsFixed(2)}s'),
    ]);
  }

  Widget _chip(String label, String value) {
    return Text.rich(TextSpan(children: [
      TextSpan(text: '$label: ', style: const TextStyle(fontSize: 12, color: Colors.grey)),
      TextSpan(text: value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
    ]));
  }
}

class _Metrics {
  final int promptTokens, completionTokens;
  final double prefillSpeed, decodeSpeed, ttft, totalTime;
  const _Metrics({
    required this.promptTokens, required this.completionTokens,
    required this.prefillSpeed, required this.decodeSpeed,
    required this.ttft, required this.totalTime,
  });
}
