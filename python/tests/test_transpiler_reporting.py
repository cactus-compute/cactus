import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from cactus.transpiler.Converter import models as CModels
from cactus.transpiler.Converter import input_processor
from cactus.transpiler.IR import models as IRModels
from cactus.transpiler.IR import simplify_ir, special_fusions
from cactus.transpiler.Generator import lowering_utils
from cactus.transpiler.Generator import lowering_basic_ops
from cactus.transpiler.ModelProfiles.profiles import GENERIC_TEXT_PROFILE
from cactus.cli import transpiler as cli_transpiler
from cactus.cli import convert as cli_convert
from cactus.cli import create_parser


class TestTranspilerReporting(unittest.TestCase):

    def test_nonfinite_attention_mask_constants_survive_ir_json_roundtrip(self):
        record = CModels.LayerRecord(
            index=0,
            name="mask_value",
            node_type="call_function",
            target="aten.scalar_tensor.default",
            args=CModels.jsonable((-math.inf,)),
            kwargs={},
            users=[],
            tensor_output_meta={"shape": [], "dtype": "torch.float16"},
            module_stack=None,
        )
        layer_map = CModels.LayerMap(
            model_name="mask-test",
            task="prefill_no_cache",
            graph_signature="",
            range_constants="",
            nodes=[record],
        )

        restored = CModels.LayerMap.model_validate_json(layer_map.model_dump_json())
        graph = IRModels.Graph.from_map(restored)

        self.assertEqual(restored.nodes[0].args, ["-Infinity"])
        value = lowering_utils.numeric_attr(graph.nodes[0], "arg_0")
        self.assertTrue(math.isinf(value))
        self.assertLess(value, 0)

    def test_float32_cast_before_pow_is_not_elided(self):
        source = mock.Mock(
            tensor_output_meta={"shape": [1, 8], "dtype": "torch.float16"},
        )
        power = mock.Mock(target="cactus.pow")
        cast = mock.Mock(parents=[source], children=[power])

        self.assertFalse(lowering_basic_ops.can_skip_float32_copy(cast))

    def test_lfm_text_prefill_slices_lm_head_before_projection(self):
        activation = mock.Mock(
            name="hidden", target="cactus.view", parents=[],
            tensor_output_meta={"shape": [128, 2048]}, module_stack=None,
        )
        weight = mock.Mock(name="lm_head_weight", target="lm_head_weight", parents=[])
        projection = mock.Mock(
            name="lm_head", target="cactus.linear", parents=[activation, weight],
            tensor_output_meta={"shape": [128, 65536]},
            module_stack=[{"module_path": "model.lm_head"}],
        )
        context = mock.Mock(component=mock.Mock(name="decoder_prefill_text_chunk"))

        self.assertTrue(lowering_basic_ops.is_decoder_prefill_logits_matmul(context, projection))

    def test_non_prefill_or_non_lm_head_projection_is_not_sliced(self):
        activation = mock.Mock(
            name="hidden", target="cactus.view", parents=[],
            tensor_output_meta={"shape": [128, 2048]}, module_stack=None,
        )
        weight = mock.Mock(name="weight", target="weight", parents=[])
        projection = mock.Mock(
            name="wide_mlp", target="cactus.linear", parents=[activation, weight],
            tensor_output_meta={"shape": [128, 65536]},
            module_stack=[{"module_path": "model.layers.0.mlp.up_proj"}],
        )

        decode = mock.Mock(component=mock.Mock(name="decoder_step"))
        prefill = mock.Mock(component=mock.Mock(name="decoder_prefill_text_chunk"))
        self.assertFalse(lowering_basic_ops.is_decoder_prefill_logits_matmul(decode, projection))
        self.assertFalse(lowering_basic_ops.is_decoder_prefill_logits_matmul(prefill, projection))

    def test_generic_text_profile_uses_processor_input(self):
        self.assertEqual(GENERIC_TEXT_PROFILE.input_strategy, "processor")

    def test_registered_model_uses_exact_profile_without_generic_guessing(self):
        resolved = cli_transpiler.resolve_transpile_config("google/gemma-4-E2B-it")

        self.assertEqual(resolved.profile_source, "registered")
        self.assertEqual(resolved.profile.model_profiles, "gemma4_e2b_it")
        self.assertEqual(resolved.modalities, ("text", "vision", "audio"))

    def test_optimized_model_variants_are_explicitly_registered(self):
        expected = {
            "openai/whisper-small": "whisper",
            "LiquidAI/LFM2-VL-450M": "lfm_vlm",
            "LiquidAI/LFM2-VL-3B": "lfm_vlm",
        }
        for model_id, profile_name in expected.items():
            with self.subTest(model_id=model_id):
                resolved = cli_transpiler.resolve_transpile_config(model_id)
                self.assertEqual(resolved.profile_source, "registered")
                self.assertEqual(resolved.profile.model_profiles, profile_name)

    def test_registered_model_rejects_generic_only_flags(self):
        with self.assertRaisesRegex(RuntimeError, "registered optimized profile"):
            cli_transpiler.resolve_transpile_config(
                "openai/whisper-tiny", input_modalities="audio,text",
            )

    def test_unknown_model_uses_only_explicit_generic_contract(self):
        resolved = cli_transpiler.resolve_transpile_config(
            "example/name-contains-whisper-vision-but-is-not-registered",
            input_modalities="text,vision",
            generic_task="causal-lm",
            cache_style="dynamic-kv",
            fusion_groups="generic,linear,attention",
        )

        self.assertEqual(resolved.profile_source, "generic")
        self.assertEqual(resolved.modalities, ("text", "vision"))
        self.assertEqual(resolved.profile.load_strategy, "image_text_to_text")
        self.assertEqual(
            resolved.profile.fusion_fields,
            ("generic", "linear", "attention", "generic_cached_attention"),
        )

    def test_unknown_model_defaults_to_generic_text_no_cache(self):
        resolved = cli_transpiler.resolve_transpile_config("example/completely-unknown")

        self.assertEqual(resolved.profile_source, "generic")
        self.assertEqual(resolved.generic_task, "causal-lm")
        self.assertEqual(resolved.cache_style, "none")
        self.assertEqual(resolved.modalities, ("text",))
        self.assertEqual(resolved.profile.cache_contract.max_cache_sequence_length, 0)
        self.assertEqual(resolved.inference_modes, ("prefill_no_cache",))
        self.assertEqual(resolved.profile.cache_policy, ("no_cache_full_context",))

    def test_unknown_text_model_can_select_full_context_no_cache_fallback(self):
        resolved = cli_transpiler.resolve_transpile_config(
            "example/completely-unknown",
            input_modalities="text",
            generic_task="causal-lm",
            cache_style="none",
        )

        self.assertEqual(resolved.profile_source, "generic")
        self.assertEqual(resolved.cache_style, "none")
        self.assertEqual(resolved.inference_modes, ("prefill_no_cache",))
        self.assertEqual(resolved.profile.cache_type, ())
        self.assertEqual(resolved.profile.cache_contract.max_cache_sequence_length, 0)
        self.assertEqual(resolved.profile.runtime_contract.states, ())
        self.assertEqual(resolved.profile.runtime_contract.execution_strategy, "full_context_recompute")
        self.assertNotIn("generic_cached_attention", resolved.profile.fusion_fields)

    def test_full_context_no_cache_input_is_padded_with_inactive_mask(self):
        input_ = CModels.Input(
            args=(),
            kwargs={
                "input_ids": CModels.torch.tensor([[4, 5, 6]], dtype=CModels.torch.long),
                "attention_mask": CModels.torch.ones((1, 3), dtype=CModels.torch.long),
            },
            modalities=("text",),
            inference_mode="prefill_no_cache",
        )

        padded = CModels.pad_no_cache_full_context_input(input_, CModels.Input, capacity=8)

        self.assertEqual(tuple(padded.kwargs["input_ids"].shape), (1, 8))
        self.assertEqual(padded.kwargs["input_ids"].tolist(), [[4, 5, 6, 0, 0, 0, 0, 0]])
        self.assertEqual(padded.kwargs["attention_mask"].tolist(), [[1, 1, 1, 0, 0, 0, 0, 0]])

    def test_convert_reports_invalid_generic_contract_without_converting_weights(self):
        args = mock.Mock(
            model_id="example/unknown",
            output_dir="/tmp/unused-cactus-test-output",
            lora=None,
            bits=4,
            token=None,
            reconvert=False,
            weights_only=False,
            input_modalities="text",
            generic_task="speech-seq2seq",
            cache_style="encoder-decoder-kv",
            fusion_groups=None,
        )
        with mock.patch("cactus.cli.model.ensure_weights") as ensure_weights:
            result = cli_convert.cmd_convert(args)

        self.assertEqual(result, 1)
        ensure_weights.assert_not_called()

    def test_convert_cli_accepts_generic_contract_flags_and_legacy_modality_alias(self):
        parser = create_parser()
        args = parser.parse_args([
            "convert", "example/unknown",
            "--input-modalities", "audio,text",
            "--task", "speech-seq2seq",
            "--cache", "encoder-decoder-kv",
            "--fusion-groups", "generic,attention",
        ])

        self.assertEqual(args.input_modalities, "audio,text")
        self.assertEqual(args.generic_task, "speech-seq2seq")
        self.assertEqual(args.cache_style, "encoder-decoder-kv")
        self.assertEqual(args.fusion_groups, "generic,attention")

    def test_processor_inputs_only_load_requested_modality_assets(self):
        processor = mock.Mock(return_value={
            "input_ids": CModels.torch.ones((1, 4), dtype=CModels.torch.long),
            "attention_mask": CModels.torch.ones((1, 4), dtype=CModels.torch.long),
        })

        with mock.patch.object(input_processor, "default_processor", return_value=processor), \
             mock.patch.object(CModels, "_load_image_asset") as load_image, \
             mock.patch.object(CModels, "_load_audio_asset") as load_audio:
            CModels.build_processor_kwargs(
                "example/text-model", ("text",), {}, "generic_text",
            )

        load_image.assert_not_called()
        load_audio.assert_not_called()
        processor.assert_called_once_with(text="Describe this input.", return_tensors="pt")

    def test_processor_inputs_load_each_requested_modality_fixture(self):
        processor = mock.Mock(return_value={
            "input_ids": CModels.torch.ones((1, 4), dtype=CModels.torch.long),
        })
        processor.image_token = None
        processor.audio_token = None
        image = object()
        audio = object()

        with mock.patch.object(input_processor, "default_processor", return_value=processor), \
             mock.patch.object(CModels, "_load_image_asset", return_value=image) as load_image, \
             mock.patch.object(CModels, "_load_audio_asset", return_value=audio) as load_audio:
            CModels.build_processor_kwargs(
                "example/multimodal-model", ("vision", "audio", "text"), {}, "generic_multimodal",
            )

        load_image.assert_called_once_with()
        load_audio.assert_called_once_with()
        processor.assert_called_once_with(
            images=image,
            audio=audio,
            sampling_rate=16000,
            text="Describe this input.",
            return_tensors="pt",
        )

    def test_default_processor_falls_back_to_auto_tokenizer(self):
        tokenizer = object()

        with mock.patch("transformers.AutoProcessor.from_pretrained", side_effect=OSError("no processor")), \
             mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer:
            result = input_processor.default_processor("example/text-model", {}, "generic_text")

        self.assertIs(result, tokenizer)
        load_tokenizer.assert_called_once()
        args, kwargs = load_tokenizer.call_args
        self.assertEqual(args, ("example/text-model",))
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["trust_remote_code"])


    def test_repeated_simplification_always_runs_two_complete_rounds(self):
        layer_map = CModels.LayerMap(
            model_name="repeat-test", task="decode_with_cache",
            graph_signature="", range_constants="", nodes=[],
        )

        with mock.patch.object(simplify_ir, "simplify", return_value=layer_map) as simplify:
            result = simplify_ir.simplify_repeated(layer_map)

        self.assertIs(result, layer_map)
        self.assertEqual(simplify.call_count, 2)

    def test_gemma_vision_attention_recovers_native_bthd_input(self):
        def record(index, name, target, parent, shape, kwargs=None):
            return CModels.LayerRecord(
                index=index, name=name,
                node_type="placeholder" if parent is None else "call_function",
                target=target, args=[] if parent is None else [{"node": parent}],
                kwargs=kwargs or {}, users=[],
                tensor_output_meta={"shape": shape, "dtype": "torch.float16"},
                module_stack=None,
            )

        layer_map = CModels.LayerMap(
            model_name="gemma4-vision-layout-test", task="prefill_with_cache",
            graph_signature="", range_constants="", nodes=[
                record(0, "q_bthd", "q_bthd", None, [1, 32, 4, 16]),
                record(1, "q_bhtd", "cactus.transpose", "q_bthd", [1, 4, 32, 16],
                       {"permutation": [0, 2, 1, 3]}),
                record(2, "q_float", "cactus.precision_cast", "q_bhtd", [1, 4, 32, 16],
                       {"dtype": "torch.float32"}),
                record(3, "q_scaled", "cactus.scalar_multiply", "q_float", [1, 4, 32, 16],
                       {"value": 1.0}),
            ],
        )
        graph = IRModels.Graph.from_map(layer_map)

        native = special_fusions.gemma4_vision_native_attention_input(graph.nodes_map["q_scaled"])

        self.assertEqual(native.name, "q_bthd")

    def test_decomposed_attention_scale_combines_query_and_key_factors(self):
        tensor_meta = {"shape": [1, 4, 32, 16], "dtype": "torch.float16"}
        layer_map = CModels.LayerMap(
            model_name="attention-scale-test", task="prefill_with_cache",
            graph_signature="", range_constants="", nodes=[
                CModels.LayerRecord(
                    index=0, name="q", node_type="placeholder", target="q",
                    args=[], kwargs={}, users=["q_scaled"], tensor_output_meta=tensor_meta,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=1, name="k", node_type="placeholder", target="k",
                    args=[], kwargs={}, users=["k_scaled"], tensor_output_meta=tensor_meta,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=2, name="q_scaled", node_type="call_function",
                    target="cactus.scalar_multiply", args=[{"node": "q"}],
                    kwargs={"value": 0.5}, users=["qk"], tensor_output_meta=tensor_meta,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=3, name="k_scaled", node_type="call_function",
                    target="cactus.scalar_multiply", args=[{"node": "k"}],
                    kwargs={"value": 0.25}, users=["qk"], tensor_output_meta=tensor_meta,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=4, name="qk", node_type="call_function", target="aten.bmm.default",
                    args=[{"node": "q_scaled"}, {"node": "k_scaled"}], kwargs={}, users=[],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
            ],
        )
        graph = IRModels.Graph.from_map(layer_map)

        scale = special_fusions.decomposed_attention_scale(graph.nodes_map["qk"])

        self.assertAlmostEqual(scale, 0.125)

    def test_cache_concat_contract_ignores_operand_order_and_layout_wrappers(self):
        def record(index, name, target, args, shape):
            return CModels.LayerRecord(
                index=index, name=name,
                node_type="placeholder" if not args else "call_function",
                target=target, args=args, kwargs={}, users=[],
                tensor_output_meta={"shape": shape, "dtype": "torch.float16"},
                module_stack=None,
            )

        layer_map = CModels.LayerMap(
            model_name="generic-cache-contract", task="decode_with_cache",
            graph_signature="", range_constants="", nodes=[
                record(0, "past_key_values_0", "past_key_values_0", [], [1, 2, 5, 8]),
                record(1, "cache_view", "cactus.view", [{"node": "past_key_values_0"}], [1, 2, 5, 8]),
                record(2, "key_new", "key_new", [], [1, 2, 1, 8]),
                record(3, "key_cat", "cactus.cat", [{"node": "key_new"}, {"node": "cache_view"}], [1, 2, 6, 8]),
            ],
        )
        graph = IRModels.Graph.from_map(layer_map)

        match = IRModels.find_cache_concat_ancestor(
            graph.nodes_map["key_cat"], "key", max_depth=1,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.state.name, "past_key_values_0")
        self.assertEqual(match.new_value.name, "key_new")
        self.assertEqual([node.name for node in match.state_wrappers], ["cache_view"])

    def test_generic_decomposed_cached_attention_fuses_by_typed_cache_structure(self):
        def record(index, name, target, args, shape, kwargs=None):
            return CModels.LayerRecord(
                index=index, name=name,
                node_type="placeholder" if not args else "call_function",
                target=target, args=args, kwargs=kwargs or {}, users=[],
                tensor_output_meta={"shape": shape, "dtype": "torch.float16"},
                module_stack=None,
            )

        q = [1, 4, 1, 8]
        kv_new = [1, 2, 1, 8]
        kv_all = [1, 2, 6, 8]
        layer_map = CModels.LayerMap(
            model_name="unknown-causal-architecture", task="decode_with_cache",
            graph_signature="", range_constants="", nodes=[
                record(0, "query", "query", [], q),
                record(1, "past_key_values_0", "past_key_values_0", [], [1, 2, 5, 8]),
                record(2, "past_key_values_1", "past_key_values_1", [], [1, 2, 5, 8]),
                record(3, "key_new", "key_new", [], kv_new),
                record(4, "value_new", "value_new", [], kv_new),
                record(5, "key_cat", "cactus.cat", [{"node": "past_key_values_0"}, {"node": "key_new"}], kv_all),
                record(6, "value_cat", "cactus.cat", [{"node": "past_key_values_1"}, {"node": "value_new"}], kv_all),
                record(7, "key_expand", "cactus.expand", [{"node": "key_cat"}], [1, 4, 6, 8]),
                record(8, "query_scaled", "cactus.scalar_multiply", [{"node": "query"}], q, {"value": 0.5}),
                record(9, "qk", "aten.bmm.default", [{"node": "query_scaled"}, {"node": "key_expand"}], [4, 1, 6]),
                record(10, "softmax", "cactus.softmax", [{"node": "qk"}], [1, 4, 1, 6], {"axis": -1}),
                record(11, "value_expand", "cactus.expand", [{"node": "value_cat"}], [1, 4, 6, 8]),
                record(12, "value_bmm", "aten.bmm.default", [{"node": "softmax"}, {"node": "value_expand"}], [4, 1, 8]),
                record(13, "attention_out", "cactus.view", [{"node": "value_bmm"}], q),
                CModels.LayerRecord(
                    index=14, name="output", node_type="output", target="output",
                    args=[[{"node": "attention_out"}, {"node": "key_cat"}, {"node": "value_cat"}]],
                    kwargs={}, users=[], tensor_output_meta=None, module_stack=None,
                ),
            ],
        )

        simplified = simplify_ir.simplify(
            layer_map,
            fusion_fields=("generic", "attention", "cache", "generic_cached_attention"),
        )
        cached = [node for node in simplified.nodes if node.target == "cactus.attention_cached"]

        self.assertEqual(len(cached), 1)
        self.assertEqual(
            [item["node"] for item in cached[0].args],
            ["query", "key_new", "value_new", "past_key_values_0", "past_key_values_1"],
        )
        self.assertEqual(cached[0].kwargs["scale"], 0.5)

    def test_noop_cleanup_exposes_fusion_in_same_simplify_call(self):
        tensor_meta = {"shape": [1, 8], "dtype": "torch.float16"}
        layer_map = CModels.LayerMap(
            model_name="cleanup-fusion-test",
            task="prefill_with_cache",
            graph_signature="",
            range_constants="",
            nodes=[
                CModels.LayerRecord(
                    index=0, name="x", node_type="placeholder", target="x",
                    args=[], kwargs={}, users=["sigmoid", "mul"],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=1, name="sigmoid", node_type="call_function", target="aten.sigmoid.default",
                    args=[{"node": "x"}], kwargs={}, users=["clone"],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=2, name="clone", node_type="call_function", target="aten.clone.default",
                    args=[{"node": "sigmoid"}], kwargs={}, users=["mul"],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=3, name="mul", node_type="call_function", target="aten.mul.Tensor",
                    args=[{"node": "x"}, {"node": "clone"}], kwargs={}, users=["output"],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=4, name="output", node_type="output", target="output",
                    args=[[{"node": "mul"}]], kwargs={}, users=[],
                    tensor_output_meta=tensor_meta, module_stack=None,
                ),
            ],
        )

        simplified = simplify_ir.simplify(layer_map, fusion_fields=("generic", "activation"))

        operations = [node for node in simplified.nodes if node.node_type == "call_function"]
        self.assertEqual([node.target for node in operations], ["cactus.silu"])

    def test_simplify_composes_consecutive_transposes(self):
        def record(index, name, target, parent, shape, permutation=None):
            kwargs = {} if permutation is None else {"permutation": list(permutation)}
            return CModels.LayerRecord(
                index=index, name=name,
                node_type="placeholder" if parent is None else "call_function",
                target=target, args=[] if parent is None else [{"node": parent}],
                kwargs=kwargs, users=[],
                tensor_output_meta={"shape": list(shape), "dtype": "torch.float16"},
                module_stack=None,
            )

        nodes = [
            record(0, "x", "x", None, (1, 2, 3, 4)),
            record(1, "t0", "cactus.transpose", "x", (3, 1, 2, 4), (2, 0, 1, 3)),
            record(2, "t1", "cactus.transpose", "t0", (1, 2, 3, 4), (1, 2, 0, 3)),
            record(3, "t2", "cactus.transpose", "t1", (1, 3, 2, 4), (0, 2, 1, 3)),
            CModels.LayerRecord(
                index=4, name="output", node_type="output", target="output",
                args=[[{"node": "t2"}]], kwargs={}, users=[],
                tensor_output_meta={"shape": [1, 3, 2, 4], "dtype": "torch.float16"},
                module_stack=None,
            ),
        ]
        layer_map = CModels.LayerMap(
            model_name="transpose-chain", task="prefill_with_cache",
            graph_signature="", range_constants="", nodes=nodes,
        )

        simplified = simplify_ir.simplify(layer_map, fusion_fields=())
        transposes = [node for node in simplified.nodes if node.target == "cactus.transpose"]

        self.assertEqual(len(transposes), 1)
        self.assertEqual(transposes[0].args, [{"node": "x"}])
        self.assertEqual(transposes[0].kwargs["permutation"], [0, 2, 1, 3])

    def test_simplify_fuses_lm_head_softcap_chain(self):
        def operation(index, name, target, parents, shape, kwargs=None, module_stack=None):
            return CModels.LayerRecord(
                index=index, name=name, node_type="call_function", target=target,
                args=[{"node": parent} for parent in parents], kwargs=kwargs or {}, users=[],
                tensor_output_meta={"shape": list(shape), "dtype": "torch.float16"},
                module_stack=module_stack,
            )

        tensor_meta = {"shape": [1, 4], "dtype": "torch.float16"}
        nodes = [
            CModels.LayerRecord(index=0, name="hidden", node_type="placeholder", target="hidden",
                                args=[], kwargs={}, users=[], tensor_output_meta=tensor_meta, module_stack=None),
            CModels.LayerRecord(index=1, name="lm_head_weight", node_type="placeholder", target="lm_head_weight",
                                args=[], kwargs={}, users=[], tensor_output_meta={"shape": [8, 4], "dtype": "torch.float16"}, module_stack=None),
            operation(2, "linear", "cactus.linear", ("hidden", "lm_head_weight"), (1, 8),
                      {"pretransposed_rhs": True}, [{"module_path": "model.lm_head"}]),
            operation(3, "view", "cactus.view", ("linear",), (1, 1, 8), {"shape": [1, 1, 8]}),
            operation(4, "divide", "cactus.scalar_divide", ("view",), (1, 1, 8), {"value": 30.0}),
            operation(5, "tanh", "cactus.tanh", ("divide",), (1, 1, 8)),
            operation(6, "softcap", "cactus.scalar_multiply", ("tanh",), (1, 1, 8), {"value": 30.0}),
            CModels.LayerRecord(index=7, name="output", node_type="output", target="output",
                                args=[[{"node": "softcap"}]], kwargs={}, users=[],
                                tensor_output_meta={"shape": [1, 1, 8], "dtype": "torch.float16"}, module_stack=None),
        ]
        layer_map = CModels.LayerMap(model_name="softcap", task="decode_with_cache",
                                     graph_signature="", range_constants="", nodes=nodes)

        simplified = simplify_ir.simplify(layer_map, fusion_fields=())
        operations = [node for node in simplified.nodes if node.node_type == "call_function"]

        self.assertEqual([node.target for node in operations], ["cactus.logits_tq_softcap"])
        self.assertEqual(operations[0].args, [{"node": "hidden"}, {"node": "lm_head_weight"}])
        self.assertEqual(operations[0].kwargs["cap"], 30.0)

    def test_direct_view_fusion_is_idempotent(self):
        input_meta = {"shape": [1, 8], "dtype": "torch.float16"}
        output_meta = {"shape": [2, 4], "dtype": "torch.float16"}
        layer_map = CModels.LayerMap(
            model_name="view-idempotence-test", task="prefill_with_cache",
            graph_signature="", range_constants="", nodes=[
                CModels.LayerRecord(
                    index=0, name="x", node_type="placeholder", target="x",
                    args=[], kwargs={}, users=["view"], tensor_output_meta=input_meta,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=1, name="view", node_type="call_function", target="aten.view.default",
                    args=[{"node": "x"}, [2, 4]], kwargs={}, users=["output"],
                    tensor_output_meta=output_meta, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=2, name="output", node_type="output", target="output",
                    args=[[{"node": "view"}]], kwargs={}, users=[],
                    tensor_output_meta=output_meta, module_stack=None,
                ),
            ],
        )

        once = simplify_ir.simplify(layer_map, fusion_fields=("generic",))
        twice = simplify_ir.simplify(once, fusion_fields=("generic",))

        self.assertEqual(once.nodes[1].target, "cactus.view")
        self.assertEqual(once.model_dump(), twice.model_dump())

    def test_decode_qkv_projections_share_one_fused_transform(self):
        hidden_meta = {"shape": [1, 1, 16], "dtype": "torch.float16"}
        stack_base = [
            {"module_path": "model.model.language_model.layers.0.self_attn", "module_type": "Attention"},
        ]
        nodes = [
            CModels.LayerRecord(index=0, name="hidden", node_type="placeholder", target="hidden",
                                args=[], kwargs={}, users=[], tensor_output_meta=hidden_meta, module_stack=None),
        ]
        outputs = []
        for offset, (role, width) in enumerate((("q", 24), ("k", 8), ("v", 8)), start=1):
            weight = f"{role}_weight"
            view = f"{role}_view"
            linear = f"{role}_linear"
            nodes.extend([
                CModels.LayerRecord(index=offset * 3 - 2, name=weight, node_type="placeholder", target=weight,
                                    args=[], kwargs={}, users=[], tensor_output_meta={"shape": [width, 16], "dtype": "torch.float16"}, module_stack=None),
                CModels.LayerRecord(index=offset * 3 - 1, name=view, node_type="call_function", target="cactus.view",
                                    args=[{"node": "hidden"}], kwargs={"shape": [1, 16]}, users=[],
                                    tensor_output_meta={"shape": [1, 16], "dtype": "torch.float16"}, module_stack=None),
                CModels.LayerRecord(index=offset * 3, name=linear, node_type="call_function", target="cactus.linear",
                                    args=[{"node": view}, {"node": weight}], kwargs={}, users=[],
                                    tensor_output_meta={"shape": [1, width], "dtype": "torch.float16"},
                                    module_stack=[*stack_base, {"module_path": f"model.model.language_model.layers.0.self_attn.{role}_proj", "module_type": "Linear"}]),
            ])
            outputs.append({"node": linear})
        nodes.append(CModels.LayerRecord(index=20, name="output", node_type="output", target="output",
                                         args=[outputs], kwargs={}, users=[], tensor_output_meta=None, module_stack=None))
        layer_map = CModels.LayerMap(model_name="google/gemma-4-E2B-it", task="decode_with_cache",
                                     graph_signature="", range_constants="", nodes=nodes)

        simplified = simplify_ir.simplify(layer_map, fusion_fields=("decode_qkv",))
        targets = [node.target for node in simplified.nodes]

        self.assertEqual(targets.count("cactus.qkv_tq_fused"), 1)
        self.assertEqual(targets.count("cactus.linear"), 0)
        self.assertEqual(targets.count("cactus.slice"), 3)

    def test_lfm_decode_w1_w3_share_one_fused_transform(self):
        hidden_meta = {"shape": [1, 1, 16], "dtype": "torch.float16"}
        nodes = [CModels.LayerRecord(
            index=0, name="hidden", node_type="placeholder", target="hidden", args=[], kwargs={}, users=[],
            tensor_output_meta=hidden_meta, module_stack=None,
        )]
        outputs = []
        for offset, role in enumerate(("w1", "w3"), start=1):
            nodes.extend([
                CModels.LayerRecord(
                    index=offset * 3 - 2, name=f"{role}_weight", node_type="placeholder", target=f"{role}_weight",
                    args=[], kwargs={}, users=[], tensor_output_meta={"shape": [32, 16], "dtype": "torch.float16"}, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=offset * 3 - 1, name=f"{role}_view", node_type="call_function", target="cactus.view",
                    args=[{"node": "hidden"}], kwargs={"shape": [1, 16]}, users=[],
                    tensor_output_meta={"shape": [1, 16], "dtype": "torch.float16"}, module_stack=None,
                ),
                CModels.LayerRecord(
                    index=offset * 3, name=f"{role}_linear", node_type="call_function", target="cactus.linear",
                    args=[{"node": f"{role}_view"}, {"node": f"{role}_weight"}], kwargs={}, users=[],
                    tensor_output_meta={"shape": [1, 32], "dtype": "torch.float16"},
                    module_stack=[{"module_path": f"model.layers.0.feed_forward.{role}", "module_type": "Linear"}],
                ),
            ])
            outputs.append({"node": f"{role}_linear"})
        nodes.append(CModels.LayerRecord(
            index=10, name="output", node_type="output", target="output", args=[outputs], kwargs={}, users=[],
            tensor_output_meta=None, module_stack=None,
        ))
        layer_map = CModels.LayerMap(
            model_name="LiquidAI/LFM", task="decode_with_cache", graph_signature="", range_constants="", nodes=nodes,
        )

        simplified = simplify_ir.simplify(layer_map, fusion_fields=("decode_projection_pair",))
        targets = [node.target for node in simplified.nodes]

        self.assertEqual(targets.count("cactus.projection_pair_tq_fused"), 1)
        self.assertEqual(targets.count("cactus.linear"), 0)
        self.assertEqual(targets.count("cactus.slice"), 2)

    def test_simplified_ir_writes_operation_and_fusion_report(self):
        layer_map = CModels.LayerMap(
            model_name="report-test",
            task="decode_with_cache",
            graph_signature="",
            range_constants="",
            nodes=[
                CModels.LayerRecord(
                    index=0, name="x", node_type="placeholder", target="x",
                    args=[], kwargs={}, users=["neg"], tensor_output_meta=None,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=1, name="neg", node_type="call_function", target="aten.neg.default",
                    args=[{"node": "x"}], kwargs={}, users=["output"], tensor_output_meta=None,
                    module_stack=None,
                ),
                CModels.LayerRecord(
                    index=2, name="output", node_type="output", target="output",
                    args=[[{"node": "neg"}]], kwargs={}, users=[], tensor_output_meta=None,
                    module_stack=None,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            simplified_path = Path(tmpdir) / "model.simplified.json"
            simplify_ir.write_simplified_json(layer_map, simplified_path)
            report_path = simplified_path.with_suffix(".fusion_report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["model_name"], "report-test")
        self.assertEqual(report["before"]["operations"]["aten.neg.default"], 1)
        self.assertIn("applied_fusion_counts", report)
        self.assertIn("missed_candidates", report)


if __name__ == "__main__":
    unittest.main()
