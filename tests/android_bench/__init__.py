"""
Android benchmarking harness for the Cactus inference engine.

Extends the Cactus performance table with broader Android device coverage,
INT4/INT8 precision comparison, and new battery + thermal throttling metrics
that no other mobile inference benchmark currently tracks.

Usage:
    cd tests/android_bench
    python bench.py --serial <adb_serial>
"""
