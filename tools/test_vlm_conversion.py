#!/usr/bin/env python3
"""
Test script for VLM weight conversion
Tests architecture detection and weight pattern matching without downloading full models
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vlm_architectures import detect_vlm_architecture, get_vision_encoder_patterns, get_projection_patterns
except ImportError:
    print("Error: Could not import vlm_architectures module")
    sys.exit(1)


class MockConfig:
    """Mock configuration object for testing"""
    def __init__(self, model_type, has_vision=True):
        self.model_type = model_type
        self.text_config = type('obj', (object,), {'model_type': model_type})()
        if has_vision:
            self.vision_config = type('obj', (object,), {'hidden_size': 1024})()
        else:
            self.vision_config = None


def test_architecture_detection():
    """Test VLM architecture detection"""
    print("Testing Architecture Detection...")
    print("=" * 60)
    
    test_cases = [
        ("smolvlm", "smolvlm"),
        ("smol_vlm", "smolvlm"),
        ("qwen_vl", "qwen_vl"),
        ("qwen-vl-chat", "qwen_vl"),
        ("llava", "llava"),
        ("llava-v1.5", "llava"),
        ("unknown_model", "smolvlm"),  # Should default to smolvlm
    ]
    
    passed = 0
    failed = 0
    
    for model_type, expected in test_cases:
        config = MockConfig(model_type)
        detected = detect_vlm_architecture(config, model_type)
        
        if detected == expected:
            print(f"✓ {model_type:20s} -> {detected:15s} (expected: {expected})")
            passed += 1
        else:
            print(f"✗ {model_type:20s} -> {detected:15s} (expected: {expected})")
            failed += 1
    
    print(f"\nArchitecture Detection: {passed} passed, {failed} failed")
    return failed == 0


def test_vision_encoder_patterns():
    """Test vision encoder weight patterns"""
    print("\nTesting Vision Encoder Patterns...")
    print("=" * 60)
    
    architectures = ["smolvlm", "qwen_vl", "llava"]
    
    for arch in architectures:
        print(f"\n{arch.upper()}:")
        patterns = get_vision_encoder_patterns(arch)
        
        # Check required keys
        required_keys = ['embeddings', 'layer_prefix_pattern', 'layer_weights']
        missing_keys = [k for k in required_keys if k not in patterns]
        
        if missing_keys:
            print(f"  ✗ Missing required keys: {missing_keys}")
            return False
        
        print(f"  ✓ Embeddings: {len(patterns['embeddings'])} patterns")
        print(f"  ✓ Layer pattern: {patterns['layer_prefix_pattern']}")
        print(f"  ✓ Layer weights: {list(patterns['layer_weights'].keys())}")
        
        # Check layer weight categories
        layer_weights = patterns['layer_weights']
        for category in ['norm1', 'norm2', 'mlp', 'attn']:
            if category in layer_weights:
                print(f"    - {category}: {len(layer_weights[category])} patterns")
    
    print("\n✓ All vision encoder patterns valid")
    return True


def test_projection_patterns():
    """Test projection layer patterns"""
    print("\nTesting Projection Patterns...")
    print("=" * 60)
    
    architectures = ["smolvlm", "qwen_vl", "llava"]
    
    for arch in architectures:
        patterns = get_projection_patterns(arch)
        print(f"\n{arch.upper()}: {len(patterns)} projection patterns")
        
        for key, outname in patterns[:3]:  # Show first 3
            print(f"  - {key:50s} -> {outname}")
        
        if len(patterns) > 3:
            print(f"  ... and {len(patterns) - 3} more")
    
    print("\n✓ All projection patterns valid")
    return True


def test_weight_pattern_format():
    """Test that weight patterns are properly formatted"""
    print("\nTesting Weight Pattern Format...")
    print("=" * 60)
    
    architectures = ["smolvlm", "qwen_vl", "llava"]
    
    for arch in architectures:
        patterns = get_vision_encoder_patterns(arch)
        layer_weights = patterns['layer_weights']
        
        # Test that layer weight patterns can be formatted
        for category, weight_list in layer_weights.items():
            for rel_name, out_template in weight_list:
                if out_template is not None:
                    try:
                        # Test formatting with layer index
                        formatted = out_template.format(0)
                        if not formatted.endswith('.weights'):
                            print(f"  ✗ {arch}/{category}: Output name doesn't end with .weights: {formatted}")
                            return False
                    except Exception as e:
                        print(f"  ✗ {arch}/{category}: Failed to format template: {e}")
                        return False
        
        print(f"  ✓ {arch}: All weight patterns properly formatted")
    
    print("\n✓ All weight pattern formats valid")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VLM Weight Conversion Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Architecture Detection", test_architecture_detection),
        ("Vision Encoder Patterns", test_vision_encoder_patterns),
        ("Projection Patterns", test_projection_patterns),
        ("Weight Pattern Format", test_weight_pattern_format),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:10s} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
