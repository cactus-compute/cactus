#!/usr/bin/env python3
"""
Example: Vision-Language Model with Tool Calling

This example demonstrates how to use Cactus VLM models with tool calling
to get structured outputs from image analysis.

Use case: Analyze an image and get structured JSON output with description
and category classification.
"""

import json
from pathlib import Path
from cactus import cactus_init, cactus_complete, cactus_destroy

# Paths
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets"


def main():
    print("=" * 60)
    print("VLM with Tool Calling Example")
    print("=" * 60)
    
    # Load VLM model
    model_path = str(WEIGHTS_DIR / "lfm2-vl-450m")
    print(f"\n📦 Loading VLM model from: {model_path}")
    
    vlm = cactus_init(model_path)
    if not vlm:
        print("❌ Failed to load model. Make sure you've downloaded it:")
        print("   cactus download LiquidAI/LFM2-VL-450M")
        return
    
    print("✅ Model loaded successfully!")
    
    # Define tool for structured image analysis
    tools = [{
        "type": "function",
        "function": {
            "name": "analyze_image_output",
            "description": "Analyze image content and provide structured output with description and category",
            "parameters": {
                "type": "object",
                "properties": {
                    "desc": {
                        "type": "string",
                        "description": "A detailed description of the image content, including people, objects, scenes, colors, actions, and other details visible in the image"
                    },
                    "category": {
                        "type": "string",
                        "description": "The primary category of the image",
                        "enum": [
                            "animals",
                            "characters",
                            "cartoons",
                            "nature",
                            "objects",
                            "scenes",
                            "food",
                            "technology"
                        ]
                    }
                },
                "required": ["desc", "category"]
            }
        }
    }]
    
    # Prepare message with image
    image_path = str(ASSETS_DIR / "test_monkey.png")
    print(f"\n🖼️  Analyzing image: {image_path}")
    
    messages = [{
        "role": "user",
        "content": "Please analyze this image and provide a detailed description with category.",
        "images": [image_path]
    }]
    
    # Call VLM with tool calling
    print("\n🔄 Processing with tool calling...")
    
    response_json = cactus_complete(
        vlm,
        json.dumps(messages),
        tools=json.dumps(tools)
    )
    
    # Parse response
    try:
        result = json.loads(response_json)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse response: {e}")
        print(f"Raw response: {response_json}")
        cactus_destroy(vlm)
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Regular response (if any)
    if result.get("response"):
        print(f"\n💬 Response: {result['response']}")
    
    # Function calls (structured output)
    function_calls = result.get("function_calls", [])
    if function_calls:
        print(f"\n🔧 Function Calls: {len(function_calls)}")
        for i, call in enumerate(function_calls, 1):
            print(f"\n  Call #{i}:")
            if isinstance(call, str):
                # Parse if it's a JSON string
                try:
                    call = json.loads(call)
                except:
                    pass
            
            if isinstance(call, dict):
                print(f"    Function: {call.get('name', 'unknown')}")
                args = call.get('arguments', {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                
                if isinstance(args, dict):
                    print(f"    Description: {args.get('desc', 'N/A')}")
                    print(f"    Category: {args.get('category', 'N/A')}")
                else:
                    print(f"    Arguments: {args}")
            else:
                print(f"    {call}")
    else:
        print("\n⚠️  No function calls returned")
    
    # Performance metrics
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"\n📊 Performance:")
        print(f"    TTFT: {metrics.get('ttft_ms', 0):.1f}ms")
        print(f"    Decode: {metrics.get('decode_tps', 0):.1f} tok/s")
        print(f"    Tokens: {metrics.get('prompt_tokens', 0)} + {metrics.get('completion_tokens', 0)}")
    
    print("\n" + "=" * 60)
    
    # Cleanup
    cactus_destroy(vlm)
    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
