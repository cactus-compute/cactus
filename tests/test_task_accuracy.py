#!/usr/bin/env python3
"""
Phase 1f: Task-Specific Accuracy Regression

Evaluates INT4 KV cache on structured benchmarks that stress different model
capabilities. Catches failure modes that perplexity misses — a model might
maintain low perplexity on prose while severely degrading on code or math.

Three benchmark suites:
  1. Code generation (20 problems): syntactic validity, function definition presence
  2. Math reasoning (15 GSM8K-style problems): correct final answer extraction
  3. Instruction following (10 problems): format compliance (JSON, lists, constraints)

Modes:
  --save-baseline FILE   Run all benchmarks and save results as baseline JSON
  --baseline FILE        Compare current results against saved baseline, report regressions
  (no flag)              Run benchmarks and report metrics without comparison

Usage:
  python tests/test_task_accuracy.py --model weights/Qwen3-0.6B
  python tests/test_task_accuracy.py --model weights/Qwen3-0.6B --save-baseline results/baseline_int8.json
  python tests/test_task_accuracy.py --model weights/Qwen3-0.6B --baseline results/baseline_int8.json
"""
import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))
import ctypes
from cactus import cactus_init, cactus_reset, cactus_destroy, _lib, TokenCallback


def cactus_complete_raw(model, messages, options):
    """Call cactus_complete with arbitrary options JSON (supports disable_thinking)."""
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback()
    msgs_json = json.dumps(messages) if isinstance(messages, list) else messages
    opts_json = json.dumps(options) if isinstance(options, dict) else options
    _lib.cactus_complete(model, msgs_json.encode(), buf, len(buf), opts_json.encode(), None, cb, None)
    return buf.value.decode("utf-8", errors="ignore")


INFERENCE_OPTIONS = {
    "temperature": 0,
    "top_k": 1,
    "max_tokens": 256,
    "disable_thinking": True,
}


def strip_thinking(response: str) -> str:
    """Strip <think>...</think> blocks from model responses (Qwen3 thinking mode).
    Also handles unclosed <think> blocks (model ran out of tokens mid-thought)."""
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    stripped = re.sub(r"<think>.*", "", response, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    return response.strip()


# ---------------------------------------------------------------------------
# Benchmark suite 1: Code Generation (20 problems)
# ---------------------------------------------------------------------------

CODE_PROBLEMS = [
    {
        "id": "code_01",
        "prompt": "Write a Python function called `fibonacci` that takes an integer n and returns the nth Fibonacci number (0-indexed, so fibonacci(0)=0, fibonacci(1)=1, fibonacci(6)=8). Only output the function, no explanation.",
        "check_fn_name": "fibonacci",
    },
    {
        "id": "code_02",
        "prompt": "Write a Python function called `fizzbuzz` that takes an integer n and returns a list of strings from 1 to n where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', and other numbers are their string representation. Only output the function, no explanation.",
        "check_fn_name": "fizzbuzz",
    },
    {
        "id": "code_03",
        "prompt": "Write a Python function called `reverse_string` that takes a string and returns it reversed. Only output the function, no explanation.",
        "check_fn_name": "reverse_string",
    },
    {
        "id": "code_04",
        "prompt": "Write a Python function called `is_palindrome` that takes a string and returns True if it reads the same forwards and backwards (case-insensitive, ignoring spaces), False otherwise. Only output the function, no explanation.",
        "check_fn_name": "is_palindrome",
    },
    {
        "id": "code_05",
        "prompt": "Write a Python function called `find_max` that takes a list of numbers and returns the maximum value without using the built-in max function. Only output the function, no explanation.",
        "check_fn_name": "find_max",
    },
    {
        "id": "code_06",
        "prompt": "Write a Python function called `bubble_sort` that takes a list of numbers and returns a new sorted list using the bubble sort algorithm. Only output the function, no explanation.",
        "check_fn_name": "bubble_sort",
    },
    {
        "id": "code_07",
        "prompt": "Write a Python function called `binary_search` that takes a sorted list and a target value, and returns the index of the target or -1 if not found. Only output the function, no explanation.",
        "check_fn_name": "binary_search",
    },
    {
        "id": "code_08",
        "prompt": "Write a Python function called `factorial` that takes a non-negative integer n and returns n! (factorial of n). Only output the function, no explanation.",
        "check_fn_name": "factorial",
    },
    {
        "id": "code_09",
        "prompt": "Write a Python function called `count_vowels` that takes a string and returns the number of vowels (a, e, i, o, u, case-insensitive). Only output the function, no explanation.",
        "check_fn_name": "count_vowels",
    },
    {
        "id": "code_10",
        "prompt": "Write a Python function called `remove_duplicates` that takes a list and returns a new list with duplicates removed, preserving the original order. Only output the function, no explanation.",
        "check_fn_name": "remove_duplicates",
    },
    {
        "id": "code_11",
        "prompt": "Write a Python function called `is_prime` that takes an integer and returns True if it is a prime number, False otherwise. Only output the function, no explanation.",
        "check_fn_name": "is_prime",
    },
    {
        "id": "code_12",
        "prompt": "Write a Python function called `sum_digits` that takes a non-negative integer and returns the sum of its digits. Only output the function, no explanation.",
        "check_fn_name": "sum_digits",
    },
    {
        "id": "code_13",
        "prompt": "Write a Python function called `flatten` that takes a nested list (e.g. [[1,2],[3,[4,5]]]) and returns a flat list of all elements. Only output the function, no explanation.",
        "check_fn_name": "flatten",
    },
    {
        "id": "code_14",
        "prompt": "Write a Python function called `second_largest` that takes a list of at least 2 distinct numbers and returns the second largest value. Only output the function, no explanation.",
        "check_fn_name": "second_largest",
    },
    {
        "id": "code_15",
        "prompt": "Write a Python function called `are_anagrams` that takes two strings and returns True if they are anagrams (same letters, same count, case-insensitive), False otherwise. Only output the function, no explanation.",
        "check_fn_name": "are_anagrams",
    },
    {
        "id": "code_16",
        "prompt": "Write a Python function called `merge_sorted` that takes two sorted lists of numbers and returns a single merged sorted list. Only output the function, no explanation.",
        "check_fn_name": "merge_sorted",
    },
    {
        "id": "code_17",
        "prompt": "Write a Python function called `to_binary` that takes a non-negative integer and returns its binary representation as a string (without '0b' prefix). to_binary(0) should return '0'. Only output the function, no explanation.",
        "check_fn_name": "to_binary",
    },
    {
        "id": "code_18",
        "prompt": "Write a Python function called `gcd` that takes two positive integers and returns their greatest common divisor. Only output the function, no explanation.",
        "check_fn_name": "gcd",
    },
    {
        "id": "code_19",
        "prompt": "Write a Python function called `rotate_list` that takes a list and an integer k, and returns the list rotated right by k positions. For example, rotate_list([1,2,3,4,5], 2) returns [4,5,1,2,3]. Only output the function, no explanation.",
        "check_fn_name": "rotate_list",
    },
    {
        "id": "code_20",
        "prompt": "Write a Python function called `pascal_row` that takes a non-negative integer n and returns the nth row (0-indexed) of Pascal's triangle as a list. pascal_row(0)=[1], pascal_row(4)=[1,4,6,4,1]. Only output the function, no explanation.",
        "check_fn_name": "pascal_row",
    },
]


def extract_python_code(response: str) -> str:
    """Extract Python code from a model response, handling markdown fences."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or in_code:
            in_code = True
            code_lines.append(line)
        elif in_code and stripped == "":
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return response.strip()


def check_syntax_valid(code: str) -> bool:
    """Check if the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_has_function(code: str, fn_name: str) -> bool:
    """Check if the code defines a function with the given name."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                return True
    except SyntaxError:
        pass
    return False


def evaluate_code_problem(response: str, problem: dict) -> dict:
    code = extract_python_code(strip_thinking(response))
    syntax_ok = check_syntax_valid(code)
    has_fn = check_has_function(code, problem["check_fn_name"])
    return {
        "id": problem["id"],
        "syntax_valid": syntax_ok,
        "has_function": has_fn,
        "pass": syntax_ok and has_fn,
        "extracted_code": code,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Benchmark suite 2: Math Reasoning (15 GSM8K-style problems)
# ---------------------------------------------------------------------------

MATH_PROBLEMS = [
    {
        "id": "math_01",
        "prompt": "Sarah has 5 apples. She buys 3 more and then gives 2 away. How many apples does she have? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 6,
    },
    {
        "id": "math_02",
        "prompt": "A store sells shirts for $15 each. John buys 4 shirts and pays with a $100 bill. How much change does he receive? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 40,
    },
    {
        "id": "math_03",
        "prompt": "A train travels at 60 miles per hour. How far does it travel in 2 hours and 30 minutes? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 150,
    },
    {
        "id": "math_04",
        "prompt": "A baker makes 12 cookies per batch. She needs 84 cookies for a party. How many batches does she need to bake? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 7,
    },
    {
        "id": "math_05",
        "prompt": "Emma reads 25 pages per day. Her book has 350 pages. How many days will it take her to finish the book? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 14,
    },
    {
        "id": "math_06",
        "prompt": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area in square centimeters? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 40,
    },
    {
        "id": "math_07",
        "prompt": "Tom has 3 times as many marbles as Jerry. Jerry has 12 marbles. How many marbles do they have together? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 48,
    },
    {
        "id": "math_08",
        "prompt": "A parking lot has 4 rows with 15 spaces each. If 38 spaces are occupied, how many spaces are empty? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 22,
    },
    {
        "id": "math_09",
        "prompt": "Lisa earns $12 per hour. She worked 8 hours on Monday and 6 hours on Tuesday. How much did she earn in total? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 168,
    },
    {
        "id": "math_10",
        "prompt": "A bag contains 5 red balls, 3 blue balls, and 7 green balls. How many balls are in the bag? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 15,
    },
    {
        "id": "math_11",
        "prompt": "A school has 480 students divided equally into 16 classrooms. How many students are in each classroom? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 30,
    },
    {
        "id": "math_12",
        "prompt": "Mark buys 3 notebooks at $4 each and 2 pens at $3 each. How much does he spend in total? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 18,
    },
    {
        "id": "math_13",
        "prompt": "A car uses 5 liters of fuel per 100 km. How many liters does it need for a 340 km trip? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 17,
    },
    {
        "id": "math_14",
        "prompt": "A farmer has 156 eggs. He packs them into boxes of 12. How many full boxes can he fill? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 13,
    },
    {
        "id": "math_15",
        "prompt": "There are 5 people in a room. Each person shakes hands with every other person exactly once. How many handshakes occur in total? Show your reasoning step by step, then give the final answer as a single number on the last line.",
        "answer": 10,
    },
]


def extract_final_number(response: str) -> float | None:
    """Extract the last number from a response (the final answer)."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    if not numbers:
        return None
    return float(numbers[-1])


def evaluate_math_problem(response: str, problem: dict) -> dict:
    extracted = extract_final_number(strip_thinking(response))
    expected = problem["answer"]
    correct = False
    if extracted is not None:
        correct = abs(extracted - expected) < 0.01
    return {
        "id": problem["id"],
        "expected": expected,
        "extracted": extracted,
        "correct": correct,
        "pass": correct,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Benchmark suite 3: Instruction Following (10 problems)
# ---------------------------------------------------------------------------

INSTRUCTION_PROBLEMS = [
    {
        "id": "inst_01",
        "prompt": "List exactly 5 fruits. Output only the list, one fruit per line, nothing else.",
        "check": "line_count",
        "expected_count": 5,
    },
    {
        "id": "inst_02",
        "prompt": 'Respond with a valid JSON object that has exactly two keys: "name" (a string) and "age" (a number). Output only the JSON, nothing else.',
        "check": "json_keys",
        "expected_keys": ["name", "age"],
    },
    {
        "id": "inst_03",
        "prompt": "Write exactly 3 sentences about the ocean. Each sentence must end with a period.",
        "check": "sentence_count",
        "expected_count": 3,
    },
    {
        "id": "inst_04",
        "prompt": "Is 7 a prime number? Answer with only a single word: 'yes' or 'no'. Output nothing else.",
        "check": "exact_word",
        "expected_words": ["yes"],
    },
    {
        "id": "inst_05",
        "prompt": "Write a numbered list with exactly 4 items about benefits of exercise. Format: '1. ...', '2. ...', etc.",
        "check": "numbered_list",
        "expected_count": 4,
    },
    {
        "id": "inst_06",
        "prompt": "Output the numbers 1 through 5, separated by commas, with no spaces. Nothing else.",
        "check": "exact_match",
        "expected": "1,2,3,4,5",
    },
    {
        "id": "inst_07",
        "prompt": "Write a single word that means 'happy'. Output only one word, nothing else.",
        "check": "single_word",
    },
    {
        "id": "inst_08",
        "prompt": 'Respond with a valid JSON array containing exactly 3 strings. Output only the JSON array, nothing else.',
        "check": "json_array",
        "expected_length": 3,
    },
    {
        "id": "inst_09",
        "prompt": "What is 2+2? Reply with only the number. No words, no punctuation, just the digit.",
        "check": "exact_match",
        "expected": "4",
    },
    {
        "id": "inst_10",
        "prompt": "Write exactly 2 bullet points about cats. Use '- ' to start each bullet point.",
        "check": "bullet_count",
        "expected_count": 2,
    },
]


def evaluate_instruction_problem(response: str, problem: dict) -> dict:
    check = problem["check"]
    text = strip_thinking(response)
    passed = False
    detail = ""

    if check == "line_count":
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        passed = len(lines) == problem["expected_count"]
        detail = f"got {len(lines)} lines, expected {problem['expected_count']}"

    elif check == "json_keys":
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                expected = set(problem["expected_keys"])
                actual = set(obj.keys())
                passed = expected == actual
                detail = f"keys: {sorted(actual)}, expected: {sorted(expected)}"
            else:
                detail = f"not a JSON object, got {type(obj).__name__}"
        except json.JSONDecodeError as e:
            detail = f"invalid JSON: {e}"

    elif check == "sentence_count":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s for s in sentences if s.strip()]
        passed = len(sentences) == problem["expected_count"]
        detail = f"got {len(sentences)} sentences, expected {problem['expected_count']}"

    elif check == "exact_word":
        word = text.lower().strip().rstrip(".")
        passed = word in [w.lower() for w in problem["expected_words"]]
        detail = f"got '{word}', expected one of {problem['expected_words']}"

    elif check == "numbered_list":
        items = re.findall(r"^\d+\.\s+", text, re.MULTILINE)
        passed = len(items) == problem["expected_count"]
        detail = f"got {len(items)} numbered items, expected {problem['expected_count']}"

    elif check == "exact_match":
        passed = text.strip() == problem["expected"]
        detail = f"got '{text.strip()}', expected '{problem['expected']}'"

    elif check == "single_word":
        words = text.split()
        passed = len(words) == 1 and words[0].isalpha()
        detail = f"got {len(words)} word(s): '{text}'"

    elif check == "json_array":
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                all_str = all(isinstance(x, str) for x in arr)
                passed = len(arr) == problem["expected_length"] and all_str
                detail = f"got array of {len(arr)} items, all strings: {all_str}"
            else:
                detail = f"not a JSON array, got {type(arr).__name__}"
        except json.JSONDecodeError as e:
            detail = f"invalid JSON: {e}"

    elif check == "bullet_count":
        bullets = re.findall(r"^- ", text, re.MULTILINE)
        passed = len(bullets) == problem["expected_count"]
        detail = f"got {len(bullets)} bullets, expected {problem['expected_count']}"

    return {
        "id": problem["id"],
        "check": check,
        "pass": passed,
        "detail": detail,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(model, suite_name: str, problems: list, evaluate_fn, system_prompt: str | None = None) -> list:
    results = []
    for i, problem in enumerate(problems):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": problem["prompt"]})

        cactus_reset(model)
        raw = cactus_complete_raw(model, messages, INFERENCE_OPTIONS)

        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            resp = {"success": False, "response": None, "error": raw}

        response_text = resp.get("response") or ""
        if not resp.get("success", False):
            result = {
                "id": problem.get("id", f"{suite_name}_{i}"),
                "pass": False,
                "error": resp.get("error", "inference failed"),
                "raw_response": response_text,
            }
        else:
            result = evaluate_fn(response_text, problem)
            result["decode_tps"] = resp.get("decode_tps", 0)
            result["total_ms"] = resp.get("total_time_ms", 0)

        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {result['id']}")
        results.append(result)

    return results


def compute_suite_metrics(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0,
    }


def compare_to_baseline(current: dict, baseline: dict) -> list:
    """Compare current results to baseline, return list of regressions.
    Only compares suites that were actually run (present in current)."""
    regressions = []

    for suite_name in ["code", "math", "instruction"]:
        if suite_name not in current:
            continue
        curr_suite = current.get(suite_name, {})
        base_suite = baseline.get(suite_name, {})
        curr_metrics = curr_suite.get("metrics", {})
        base_metrics = base_suite.get("metrics", {})

        curr_rate = curr_metrics.get("pass_rate", 0)
        base_rate = base_metrics.get("pass_rate", 0)

        if curr_rate < base_rate - 0.01:
            regressions.append({
                "suite": suite_name,
                "type": "pass_rate_drop",
                "baseline": base_rate,
                "current": curr_rate,
                "delta": curr_rate - base_rate,
            })

        curr_results = {r["id"]: r for r in curr_suite.get("results", [])}
        base_results = {r["id"]: r for r in base_suite.get("results", [])}
        for pid, base_r in base_results.items():
            curr_r = curr_results.get(pid)
            if curr_r and base_r.get("pass") and not curr_r.get("pass"):
                regressions.append({
                    "suite": suite_name,
                    "type": "individual_regression",
                    "problem_id": pid,
                    "baseline_pass": True,
                    "current_pass": False,
                })

    return regressions


def strip_raw_responses(results_data: dict) -> dict:
    """Strip verbose raw_response and extracted_code fields for cleaner baseline files."""
    stripped = {}
    for key, val in results_data.items():
        if key == "results" and isinstance(val, list):
            stripped[key] = []
            for r in val:
                clean = {k: v for k, v in r.items() if k not in ("raw_response", "extracted_code")}
                stripped[key].append(clean)
        elif isinstance(val, dict):
            stripped[key] = strip_raw_responses(val)
        else:
            stripped[key] = val
    return stripped


def main():
    parser = argparse.ArgumentParser(description="Phase 1f: Task-Specific Accuracy Regression")
    parser.add_argument("--model", required=True, help="Path to model weights directory")
    parser.add_argument("--save-baseline", metavar="FILE", help="Save results as baseline JSON")
    parser.add_argument("--baseline", metavar="FILE", help="Compare against baseline JSON")
    parser.add_argument("--kv-precision", default=None, help="KV cache precision (future: int4, nf4, asymmetric)")
    parser.add_argument("--suite", choices=["code", "math", "instruction", "all"], default="all",
                        help="Run specific suite or all (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print full responses")
    args = parser.parse_args()

    if args.kv_precision:
        os.environ["CACTUS_KV_PRECISION"] = args.kv_precision

    print(f"Loading model: {args.model}")
    model = cactus_init(args.model)
    if not model:
        print("ERROR: Failed to load model")
        sys.exit(1)

    all_results = {
        "model": args.model,
        "kv_precision": args.kv_precision or "int8",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    suites = {
        "code": (
            CODE_PROBLEMS,
            evaluate_code_problem,
            "You are a Python programmer. Write clean, correct Python code. Output only code, no explanations.",
        ),
        "math": (
            MATH_PROBLEMS,
            evaluate_math_problem,
            "You are a math tutor. Solve problems step by step. Always end with the final numerical answer on the last line.",
        ),
        "instruction": (
            INSTRUCTION_PROBLEMS,
            evaluate_instruction_problem,
            "Follow the user's instructions exactly. Be precise about formatting requirements.",
        ),
    }

    run_suites = list(suites.keys()) if args.suite == "all" else [args.suite]

    for suite_name in run_suites:
        problems, eval_fn, system_prompt = suites[suite_name]
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name} ({len(problems)} problems)")
        print(f"{'='*60}")

        results = run_benchmark(model, suite_name, problems, eval_fn, system_prompt)
        metrics = compute_suite_metrics(results)
        all_results[suite_name] = {"results": results, "metrics": metrics}

        print(f"\n  Pass rate: {metrics['passed']}/{metrics['total']} ({metrics['pass_rate']:.0%})")

    cactus_destroy(model)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pass = 0
    total_count = 0
    for suite_name in run_suites:
        m = all_results[suite_name]["metrics"]
        total_pass += m["passed"]
        total_count += m["total"]
        print(f"  {suite_name:15s}: {m['passed']:2d}/{m['total']:2d} ({m['pass_rate']:.0%})")
    print(f"  {'overall':15s}: {total_pass:2d}/{total_count:2d} ({total_pass/total_count:.0%})" if total_count else "")

    if args.save_baseline:
        baseline_data = strip_raw_responses(all_results)
        Path(args.save_baseline).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_baseline, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"\nBaseline saved to {args.save_baseline}")

    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)

        regressions = compare_to_baseline(all_results, baseline_data)
        if regressions:
            print(f"\nREGRESSIONS DETECTED ({len(regressions)}):")
            for reg in regressions:
                if reg["type"] == "pass_rate_drop":
                    print(f"  [{reg['suite']}] pass rate dropped: "
                          f"{reg['baseline']:.0%} -> {reg['current']:.0%} ({reg['delta']:+.0%})")
                elif reg["type"] == "individual_regression":
                    print(f"  [{reg['suite']}] {reg['problem_id']}: was PASS, now FAIL")
            sys.exit(1)
        else:
            print("\nNo regressions detected vs baseline.")

    if args.verbose:
        for suite_name in run_suites:
            for r in all_results[suite_name]["results"]:
                if not r["pass"]:
                    print(f"\n--- {r['id']} (FAIL) ---")
                    print(r.get("raw_response", "")[:500])


if __name__ == "__main__":
    main()
