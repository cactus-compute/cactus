typedef OnInitProgress = void Function(double? progress, String statusMessage, bool isError);

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
  final int? gpuLayers;
  final int threads;
  final bool useMmap;
  final bool useMlock;
  final bool generateEmbeddings;
  final int poolingType;
  final int normalizeEmbeddings;
  final bool useFlashAttention;
  final String? cacheTypeK;
  final String? cacheTypeV;
  final OnInitProgress? onInitProgress;

  CactusInitParams({
    this.modelPath,
    this.modelUrl,
    this.modelFilename,
    this.mmprojPath,
    this.mmprojUrl,
    this.mmprojFilename,
    this.chatTemplate,
    this.contextSize = 512,
    this.batchSize = 512,
    this.ubatchSize = 512,
    this.gpuLayers = 0,
    this.threads = 4,
    this.useMmap = true,
    this.useMlock = false,
    this.generateEmbeddings = false,
    this.poolingType = 0,
    this.normalizeEmbeddings = 1,
    this.useFlashAttention = false,
    this.cacheTypeK,
    this.cacheTypeV,
    this.onInitProgress,
  }) {
    if (modelPath == null && modelUrl == null) {
      throw ArgumentError('Either modelPath or modelUrl must be provided for the main model.');
    }
    if (modelPath != null && modelUrl != null) {
      throw ArgumentError('Cannot provide both modelPath and modelUrl. Choose one for the main model.');
    }
    if (mmprojPath != null && mmprojUrl != null) {
      throw ArgumentError('Cannot provide both mmprojPath and mmprojUrl. Choose one for the multimodal projector.');
    }
  }
}