class SimpleTimer {
  DateTime _previousTime = DateTime.now();
  
  void log(String message) {
    final now = DateTime.now();
    final delta = now.difference(_previousTime);
    
    final deltaMs = delta.inMicroseconds / 1000.0;
    
    final logMessage = '(Δ: ${deltaMs.toStringAsFixed(2)}ms) $message';
    print(logMessage);
    
    _previousTime = now;
  }
} 

final timer = SimpleTimer();