import 'dart:io';
import 'dart:convert';
import './types.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';

class TelemetryRecord {
  final String os;
  final String osVersion;
  final String framework;
  final String? projectId;
  final String? deviceId;
  final String? deviceManufacturer;
  final String? deviceModel;
  final Map<String, dynamic>? telemetryPayload;
  final Map<String, dynamic>? errorPayload;
  final String timestamp;
  final String modelFilename;
  final int? nCtx;
  final int? nGpuLayers;

  TelemetryRecord({
    required this.os,
    required this.osVersion,
    required this.framework,
    this.projectId,
    this.deviceId,
    this.deviceManufacturer,
    this.deviceModel,
    this.telemetryPayload,
    this.errorPayload,
    required this.timestamp,
    required this.modelFilename,
    this.nCtx,
    this.nGpuLayers,
  });

  Map<String, dynamic> toJson() {
    return {
      'os': os,
      'os_version': osVersion,
      'framework': framework,
      'project_id': projectId,
      'device_id': deviceId,
      'device_manufacturer': deviceManufacturer,
      'device_model': deviceModel,
      if (telemetryPayload != null) 'telemetry_payload': telemetryPayload,
      if (errorPayload != null) 'error_payload': errorPayload,
      'timestamp': timestamp,
      'model_filename': modelFilename,
      if (nCtx != null) 'n_ctx': nCtx,
      if (nGpuLayers != null) 'n_gpu_layers': nGpuLayers,
    };
  }
}

class TelemetryConfig {
  final String supabaseUrl;
  final String supabaseKey;
  final String table;

  TelemetryConfig({
    required this.supabaseUrl,
    required this.supabaseKey,
    this.table = 'telemetry',
  });
}

class CactusTelemetry {
  static CactusTelemetry? _instance;
  late TelemetryConfig _config;

  CactusTelemetry._(this._config);

  static String _getFilename(String? path) {
    if (path == null || path.isEmpty) return 'unknown';
    try {
      final uri = Uri.tryParse(path);
      if (uri != null) {
        return uri.pathSegments.isNotEmpty ? uri.pathSegments.last : 'unknown';
      }
      return path.split(Platform.pathSeparator).last;
    } catch (e) {
      return 'unknown';
    }
  }

  static void autoInit() {
    _instance ??= CactusTelemetry._(TelemetryConfig(
      supabaseUrl: 'https://vlqqczxwyaodtcdmdmlw.supabase.co',
      supabaseKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZscXFjenh3eWFvZHRjZG1kbWx3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE1MTg2MzIsImV4cCI6MjA2NzA5NDYzMn0.nBzqGuK9j6RZ6mOPWU2boAC_5H9XDs-fPpo5P3WZYbI',
    ));
  }

  static void init(TelemetryConfig config) {
    _instance ??= CactusTelemetry._(config);
  }

  static Future<void> track(Map<String, dynamic> payload, CactusInitParams options) async {
    autoInit();
    await _instance!._trackInternal(payload, options);
  }

  static Future<void> error(Object error, CactusInitParams options) async {
    autoInit();
    await _instance!._errorInternal(error, options);
  }

  static Future<(String, String, String)> _getDeviceInfo() async {
    final deviceInfoPlugin = DeviceInfoPlugin();
    String deviceId = 'unknown';
    String make = 'unknown';
    String model = 'unknown';
    if (Platform.isAndroid) {
      final AndroidDeviceInfo androidInfo = await deviceInfoPlugin.androidInfo;
      deviceId = androidInfo.id;
      make = androidInfo.manufacturer;
      model = androidInfo.model;
    } else if (Platform.isIOS) {
      final IosDeviceInfo iosInfo = await deviceInfoPlugin.iosInfo;
      deviceId = iosInfo.identifierForVendor ?? 'unknown';
      make = 'Apple';
      model = iosInfo.utsname.machine;
    }
    return (deviceId, make, model);
  }

  Future<void> _trackInternal(Map<String, dynamic> payload, CactusInitParams options) async {
    final packageInfo = await PackageInfo.fromPlatform();
    final (deviceId, make, model) = await _getDeviceInfo();

    final record = TelemetryRecord(
      os: Platform.isIOS ? 'iOS' : 'Android',
      osVersion: Platform.operatingSystemVersion,
      framework: 'flutter',
      projectId: '${packageInfo.packageName}@${packageInfo.version}',
      deviceId: deviceId,
      deviceManufacturer: make,
      deviceModel: model,
      telemetryPayload: payload,
      timestamp: DateTime.now().toIso8601String(),
      modelFilename: _getFilename(options.modelPath),
      nCtx: options.contextSize,
      nGpuLayers: options.gpuLayers,
    );

    _sendRecord(record);
  }

  Future<void> _errorInternal(Object error, CactusInitParams options) async {
    final packageInfo = await PackageInfo.fromPlatform();
    final (deviceId, make, model) = await _getDeviceInfo();
    final errorPayload = {
      'message': error.toString(),
      'type': error.runtimeType.toString(),
      if (error is CactusException) 'underlying_error': error.underlyingError?.toString(),
    };

    final record = TelemetryRecord(
      os: Platform.isIOS ? 'iOS' : 'Android',
      osVersion: Platform.operatingSystemVersion,
      framework: 'flutter',
      projectId: '${packageInfo.packageName}@${packageInfo.version}',
      deviceId: deviceId,
      deviceManufacturer: make,
      deviceModel: model,
      errorPayload: errorPayload,
      timestamp: DateTime.now().toIso8601String(),
      modelFilename: _getFilename(options.modelPath),
      nCtx: options.contextSize,
      nGpuLayers: options.gpuLayers,
    );

    _sendRecord(record);
  }

  Future<void> _sendRecord(TelemetryRecord record) async {
    try {
      final client = HttpClient();
      final uri = Uri.parse('${_config.supabaseUrl}/rest/v1/${_config.table}');
      final request = await client.postUrl(uri);
      
      request.headers.set('apikey', _config.supabaseKey);
      request.headers.set('Authorization', 'Bearer ${_config.supabaseKey}');
      request.headers.set('Content-Type', 'application/json');
      request.headers.set('Prefer', 'return=minimal');
      
      final body = jsonEncode([record.toJson()]);
      request.write(body);
      
      final response = await request.close();
      await response.drain(); 
      client.close();
    } catch (e) {
      print('Error sending record: $e');
    }
  }
} 