import numpy as np


def test_lora_target_paths_selects_projection_kernels():
    import jax.numpy as jnp
    from cactus.model.finetune import lora_target_paths

    params = {
        "stack": {"layers": {"block": {"self_attn": {
            "q_proj": {"kernel": jnp.ones((2, 8, 8))},
            "gate_proj": {"kernel": jnp.ones((2, 8, 8))},
            "q_norm": {"scale": jnp.ones((2, 8))},
        }}}, "final_norm": {"scale": jnp.ones((8,))}},
        "confidence_head": {"proj": {"kernel": jnp.ones((8, 8))}},
    }
    paths = lora_target_paths(params)
    assert {p[-2] for p in paths} == {"q_proj", "gate_proj"}


def test_init_lora_shapes_and_zero_b():
    import jax
    import jax.numpy as jnp
    from cactus.model.finetune import init_lora

    params = {"self_attn": {"v_proj": {"kernel": jnp.ones((5, 6, 7))}}}
    paths = [("self_attn", "v_proj", "kernel")]
    lora = init_lora(params, paths, rank=3, key=jax.random.PRNGKey(1))
    adapter = lora[("self_attn", "v_proj", "kernel")]
    assert adapter["A"].shape == (5, 6, 3)
    assert adapter["B"].shape == (5, 3, 7)
    assert np.allclose(np.asarray(adapter["B"]), 0.0)


def test_merge_lora_is_identity_when_b_zero():
    import jax
    import jax.numpy as jnp
    from cactus.model.finetune import init_lora, lora_target_paths, merge_lora

    params = {"self_attn": {"q_proj": {"kernel": jnp.arange(16.0).reshape(1, 4, 4)}}}
    paths = lora_target_paths(params)
    lora = init_lora(params, paths, rank=2, key=jax.random.PRNGKey(0))
    merged = merge_lora(params, lora, scale=8.0)
    np.testing.assert_allclose(
        np.asarray(merged["self_attn"]["q_proj"]["kernel"]),
        np.asarray(params["self_attn"]["q_proj"]["kernel"]), atol=1e-6)


def test_merge_lora_adds_scaled_delta():
    import jax.numpy as jnp
    from cactus.model.finetune import merge_lora

    params = {"w": {"kernel": jnp.zeros((3, 4))}}
    lora = {("w", "kernel"): {"A": jnp.ones((3, 2)), "B": jnp.ones((2, 4))}}
    merged = merge_lora(params, lora, scale=0.5)
    np.testing.assert_allclose(np.asarray(merged["w"]["kernel"]),
                               np.ones((3, 4)), atol=1e-6)
