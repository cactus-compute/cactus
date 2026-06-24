# import argparse
# import json
# from collections import Counter
# from typing import Any, Dict, Optional

# import torch
# from transformers import (
#     AutoModel,
#     AutoModelForCausalLM,
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
# )


# def jsonable(x: Any) -> Any:
#     """
#     Convert graph/export objects into JSON-safe values.
#     """
#     if isinstance(x, torch.fx.Node):
#         return {"node": x.name}

#     if isinstance(x, torch.Tensor):
#         return {
#             "shape": list(x.shape),
#             "dtype": str(x.dtype),
#             "device": str(x.device),
#             "requires_grad": bool(x.requires_grad),
#         }

#     if isinstance(x, torch.Size):
#         return list(x)

#     if isinstance(x, torch.dtype):
#         return str(x)

#     if isinstance(x, torch.device):
#         return str(x)

#     if isinstance(x, slice):
#         return {
#             "slice": {
#                 "start": jsonable(x.start),
#                 "stop": jsonable(x.stop),
#                 "step": jsonable(x.step),
#             }
#         }

#     if isinstance(x, range):
#         return list(x)

#     if isinstance(x, (list, tuple)):
#         return [jsonable(v) for v in x]

#     if isinstance(x, dict):
#         return {str(k): jsonable(v) for k, v in x.items()}

#     if isinstance(x, (str, int, float, bool)) or x is None:
#         return x

#     return repr(x)


# def aten_name(target: Any) -> str:
#     """
#     Convert torch.ops.aten.linear.default into:
#         aten.linear.default

#     For non-ATen targets, falls back to str(target).
#     """
#     schema = getattr(target, "_schema", None)

#     if schema is not None:
#         # Example:
#         # schema.name = "aten::linear"
#         # schema.overload_name = "" or "Tensor" or "int"
#         if "::" in schema.name:
#             namespace, op = schema.name.split("::", 1)
#             overload = schema.overload_name if schema.overload_name else "default"
#             return f"{namespace}.{op}.{overload}"

#     if hasattr(target, "name"):
#         try:
#             return target.name()
#         except Exception:
#             pass

#     return str(target)


# def schema_string(target: Any) -> Optional[str]:
#     schema = getattr(target, "_schema", None)
#     return str(schema) if schema is not None else None


# def extract_tensor_meta(node: torch.fx.Node) -> Optional[Any]:
#     """
#     torch.export usually stores fake tensor output metadata in node.meta["val"].
#     """
#     if "val" not in node.meta:
#         return None

#     return jsonable(node.meta["val"])


# def extract_module_stack(node: torch.fx.Node) -> Optional[Any]:
#     """
#     Sometimes torch.export nodes contain original module context here.
#     This can help later when mapping ops back to model.layers.X.self_attn, mlp, etc.
#     """
#     stack = node.meta.get("nn_module_stack", None)

#     if stack is None:
#         return None

#     out = []

#     for key, value in stack.items():
#         if isinstance(value, tuple) and len(value) >= 2:
#             module_path = value[0]
#             module_type = value[1]
#             out.append(
#                 {
#                     "key": str(key),
#                     "module_path": str(module_path),
#                     "module_type": getattr(module_type, "__name__", str(module_type)),
#                 }
#             )
#         else:
#             out.append(
#                 {
#                     "key": str(key),
#                     "value": repr(value),
#                 }
#             )

#     return out


# def make_graph_json(
#     exported: torch.export.ExportedProgram,
#     model_name: str,
#     task: str,
#     attn_implementation: str,
#     ran_decompositions: bool,
# ) -> Dict[str, Any]:
#     graph_module = exported.graph_module
#     nodes_json = []

#     for i, node in enumerate(graph_module.graph.nodes):
#         target = aten_name(node.target)

#         node_json = {
#             "index": i,
#             "name": node.name,
#             "node_type": node.op,
#             "target": target,
#             "schema": schema_string(node.target),
#             "args": jsonable(node.args),
#             "kwargs": jsonable(node.kwargs),
#             "users": [user.name for user in node.users],
#         }

#         tensor_meta = extract_tensor_meta(node)
#         if tensor_meta is not None:
#             node_json["tensor_meta"] = tensor_meta

#         module_stack = extract_module_stack(node)
#         if module_stack is not None:
#             node_json["module_stack"] = module_stack

#         nodes_json.append(node_json)

#     op_counts = Counter(
#         node["target"]
#         for node in nodes_json
#         if node["node_type"] == "call_function"
#     )

#     return {
#         "model_name": model_name,
#         "task": task,
#         "attn_implementation": attn_implementation,
#         "ran_decompositions": ran_decompositions,
#         "num_nodes": len(nodes_json),
#         "num_call_function_nodes": sum(op_counts.values()),
#         "op_counts": dict(op_counts.most_common()),
#         "graph_signature": repr(exported.graph_signature),
#         "range_constraints": repr(exported.range_constraints),
#         "nodes": nodes_json,
#     }


# def load_model(model_name: str, task: str, attn_implementation: str):
#     """
#     Load HF model with eager attention when supported.
#     """
#     model_cls_map = {
#         "base": AutoModel,
#         "causal-lm": AutoModelForCausalLM,
#         "sequence-classification": AutoModelForSequenceClassification,
#     }

#     if task not in model_cls_map:
#         raise ValueError(f"Unsupported task: {task}")

#     model_cls = model_cls_map[task]

#     try:
#         model = model_cls.from_pretrained(
#             model_name,
#             attn_implementation=attn_implementation,
#         )
#     except TypeError:
#         print(
#             f"Warning: this model class did not accept "
#             f'attn_implementation="{attn_implementation}". '
#             f"Retrying without it."
#         )
#         model = model_cls.from_pretrained(model_name)

#     model.eval()

#     # Hugging Face outputs are often ModelOutput dataclasses.
#     # Tuples are easier for torch.export to serialize.
#     if hasattr(model, "config"):
#         model.config.return_dict = False

#     # Important for GPT/LLaMA/Mistral/etc.
#     # Disables KV-cache outputs and cache-specific forward paths.
#     if hasattr(model.config, "use_cache"):
#         model.config.use_cache = False

#     return model


# def export_hf_model_to_json(
#     model_name: str,
#     output_path: str,
#     task: str = "causal-lm",
#     sample_text: str = "Hello world",
#     max_length: int = 32,
#     attn_implementation: str = "eager",
#     decompose: bool = True,
#     strict: bool = False,
# ):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model = load_model(
#         model_name=model_name,
#         task=task,
#         attn_implementation=attn_implementation,
#     )

#     inputs = tokenizer(
#         sample_text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length,
#     )

#     # Some decoder-only tokenizers do not define pad_token.
#     # Not needed for a single example, but useful if you later batch examples.
#     if tokenizer.pad_token is None and tokenizer.eos_token is not None:
#         tokenizer.pad_token = tokenizer.eos_token

#     with torch.no_grad():
#         exported = torch.export.export(
#             model,
#             args=(),
#             kwargs=dict(inputs),
#             strict=strict,
#         )

#     ran_decompositions = False

#     if decompose:
#         try:
#             exported = exported.run_decompositions()
#             ran_decompositions = True
#         except Exception as e:
#             print(f"Warning: run_decompositions() failed: {repr(e)}")
#             print("Continuing with the original exported graph.")

#     graph_json = make_graph_json(
#         exported=exported,
#         model_name=model_name,
#         task=task,
#         attn_implementation=attn_implementation,
#         ran_decompositions=ran_decompositions,
#     )

#     with open(output_path, "w") as f:
#         json.dump(graph_json, f, indent=2)

#     print(f"Saved graph JSON to: {output_path}")
#     print(f"Total nodes: {graph_json['num_nodes']}")
#     print(f"call_function nodes: {graph_json['num_call_function_nodes']}")
#     print()
#     print("Top ops:")
#     for op, count in list(graph_json["op_counts"].items())[:30]:
#         print(f"{count:5d}  {op}")

#     attention_ops = [
#         op for op in graph_json["op_counts"]
#         if "attention" in op.lower() or "scaled_dot_product" in op.lower()
#     ]

#     if attention_ops:
#         print()
#         print("Attention-like ops still present:")
#         for op in attention_ops:
#             print(f"  {op}")
#         print()
#         print(
#             "If you still see aten.scaled_dot_product_attention.default, "
#             "attention was not fully broken into matmul/softmax/matmul."
#         )


# # def parse_args():
# #     parser = argparse.ArgumentParser(
# #         description="Export a Hugging Face PyTorch model to lower-level ATen JSON."
# #     )

# #     parser.add_argument(
# #         "--model",
# #         type=str,
# #         required=True,
# #         help="HF model name, e.g. bert-base-uncased or gpt2.",
# #     )

# #     parser.add_argument(
# #         "--output",
# #         type=str,
# #         default="model_aten_graph.json",
# #         help="Output JSON path.",
# #     )

# #     parser.add_argument(
# #         "--task",
# #         type=str,
# #         default="causal-lm",
# #         choices=["base", "causal-lm", "sequence-classification"],
# #         help="Which AutoModel class to use.",
# #     )

# #     parser.add_argument(
# #         "--text",
# #         type=str,
# #         default="Hello world",
# #         help="Sample text used to trace/export the model.",
# #     )

# #     parser.add_argument(
# #         "--max-length",
# #         type=int,
# #         default=32,
# #         help="Max token length for the sample input.",
# #     )

# #     parser.add_argument(
# #         "--attn-implementation",
# #         type=str,
# #         default="eager",
# #         help='Attention backend. For graph inspection, use "eager".',
# #     )

# #     parser.add_argument(
# #         "--no-decompose",
# #         action="store_true",
# #         help="Disable exported.run_decompositions().",
# #     )

# #     parser.add_argument(
# #         "--strict",
# #         action="store_true",
# #         help="Use strict=True in torch.export.export.",
# #     )

# #     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     export_hf_model_to_json(
#         model_name=args.model,
#         output_path=args.output,
#         task=args.task,
#         sample_text=args.text,
#         max_length=args.max_length,
#         attn_implementation=args.attn_implementation,
#         decompose=not args.no_decompose,
#         strict=args.strict,
#     )