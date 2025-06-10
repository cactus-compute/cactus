import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart'; // For BenchResult

class BenchmarkView extends StatelessWidget {
  final BenchResult? benchResult;

  const BenchmarkView({super.key, this.benchResult});

  @override
  Widget build(BuildContext context) {
    if (benchResult == null) {
      return const SizedBox.shrink(); // Nothing to show if no results
    }

    const spacerSmall = SizedBox(height: 10);

    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Benchmark Results:', style: Theme.of(context).textTheme.titleLarge),
              spacerSmall,
              Text('Model: ${benchResult!.modelDesc}'),
              Text('Params: ${(benchResult!.modelNParams / 1000000).toStringAsFixed(2)}M'),
              Text('Size: ${(benchResult!.modelSize / (1024 * 1024)).toStringAsFixed(2)} MB'),
              spacerSmall,
              Text('Prompt Processing (PP):', style: Theme.of(context).textTheme.titleMedium),
              Text('  Avg: ${benchResult!.ppAvg.toStringAsFixed(2)} tokens/sec'),
              Text('  StdDev: ${benchResult!.ppStd.toStringAsFixed(2)}'),
              spacerSmall,
              Text('Text Generation (TG):', style: Theme.of(context).textTheme.titleMedium),
              Text('  Avg: ${benchResult!.tgAvg.toStringAsFixed(2)} tokens/sec'),
              Text('  StdDev: ${benchResult!.tgStd.toStringAsFixed(2)}'),
            ],
          ),
        ),
      ),
    );
  }
} 