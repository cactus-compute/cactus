import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:cactus/cactus.dart';

class CactusService {
  CactusVLM? model;
  String modelUrl =
      'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf';
  String mmProjUrl =
      'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf';

  final ValueNotifier<List<ChatMessage>> messages = ValueNotifier([]);
  final ValueNotifier<bool> isLoading = ValueNotifier(true);
  final ValueNotifier<String> status = ValueNotifier('Initializing...');
  final ValueNotifier<String?> error = ValueNotifier(null);

  String? _stagedImagePath;

  Future<String> downloadModel(
    String url, {
    Function(double, String)? onProgress,
  }) async {
    final appDocDir = await getApplicationDocumentsDirectory();
    final filename =
        url.split('/').last.isEmpty
            ? "downloaded_model.gguf"
            : url.split('/').last;
    final modelPath = '${appDocDir.path}/$filename';

    final client = HttpClient();

    if (!await File(modelPath).exists()) {
      try {
        final request = await client.getUrl(Uri.parse(url));
        final response = await request.close();

        if (response.statusCode != 200) {
          throw Exception('Failed to download model: ${response.statusCode}');
        }

        final file = File(modelPath);
        final sink = file.openWrite();

        final contentLength = response.contentLength;
        int downloaded = 0;

        await for (final chunk in response) {
          sink.add(chunk);
          downloaded += chunk.length;

          if (contentLength > 0 && onProgress != null) {
            final progress = downloaded / contentLength;
            onProgress(progress, '${(progress * 100).toStringAsFixed(1)}%');
          }
        }

        await sink.close();
      } finally {
        client.close();
      }
    } else {
    }
    return modelPath;
  }

  Future<void> initialize() async {
    try {
      status.value = 'Downloading specified model';
      String modelPath = await downloadModel(modelUrl);
      print("Downloaded ");
      print(modelPath);
      status.value = 'Downloading specified mmProj ';
      String mmProjPath = await downloadModel(mmProjUrl);
      print("Downloaded");
      print(mmProjPath);

      status.value = 'Initializing';

      model = await CactusVLM.init(
        modelPath: modelPath,
        mmprojPath: mmProjPath,
        // cactusToken: 'contact founders@cactuscompute.com for enterprise token',
        onProgress: (progress, statusText, isError) {
          status.value = statusText;
          if (isError) error.value = statusText;
        },
      );

      status.value = 'Ready!';
      isLoading.value = false;
    } on CactusException catch (e) {
      error.value = e.message;
      isLoading.value = false;
    }
  }

  Future<void> sendMessage(String text) async {
    if (model == null) return;

    isLoading.value = true;

    final userMsg = ChatMessage(role: 'user', content: text);
    messages.value = [
      ...messages.value,
      userMsg,
      ChatMessage(role: 'assistant', content: ''),
    ];

    final stopwatch = Stopwatch()..start();
    int? ttft;
    String response = '';

    try {
      final systemPrompt = ChatMessage(
        role: 'system',
        content:
            'You are a helpful AI assistant. Always provide neat, straightforward, short and relevant responses. Be concise and direct.',
      );
      final conversationMessages = [
        systemPrompt,
        ...messages.value.where((m) => m.content.isNotEmpty).toList(),
      ];

      final result = await model!.completion(
        conversationMessages,
        imagePaths: _stagedImagePath != null ? [_stagedImagePath!] : [],
        maxTokens: 200,
        // mode: "localfirst", // enterprise feature: try local, fall back to cloud if local inference fails and vice versa
        onToken: (token) {
          if (token == '<|im_end|>' || token == '</s>') return false;
          ttft ??= stopwatch.elapsedMilliseconds;
          response += token;
          _updateLastMessage(response);
          return true;
        },
      );

      stopwatch.stop();
      final totalMs = stopwatch.elapsedMilliseconds;
      final tokens = result.tokensPredicted;
      final tps = tokens > 0 && totalMs > 0 ? tokens * 1000 / totalMs : 0;

      debugPrint(
        '[PERF] TTFT: ${ttft ?? 'N/A'}ms | Total: ${totalMs}ms | Tokens: $tokens | Speed: ${tps.toStringAsFixed(1)} tok/s',
      );

      _updateLastMessage(result.text.isNotEmpty ? result.text : response);
      _stagedImagePath = null;
    } on CactusException catch (e) {
      _updateLastMessage('Error: ${e.message}');
    }

    isLoading.value = false;
  }

  void _updateLastMessage(String content) {
    final msgs = List<ChatMessage>.from(messages.value);
    if (msgs.isNotEmpty && msgs.last.role == 'assistant') {
      msgs[msgs.length - 1] = ChatMessage(role: 'assistant', content: content);
      messages.value = msgs;
    }
  }

  Future<void> addImage() async {
    final assetData = await rootBundle.load('assets/image.jpg');
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/demo_image.jpg');
    await file.writeAsBytes(assetData.buffer.asUint8List());
    _stagedImagePath = file.path;
  }

  void clearConversation() {
    messages.value = [];
    model?.rewind();
  }

  void dispose() {
    model?.dispose();
    messages.dispose();
    isLoading.dispose();
    status.dispose();
    error.dispose();
  }
}
