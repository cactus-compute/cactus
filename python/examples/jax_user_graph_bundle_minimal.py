from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle
from cactus.transpile.jax_user_graph_bundle import load_jax_user_graph_bundle


class TinyClassifier:
    def __call__(self, params, token_ids):
        embedded = params["embedding"][token_ids]
        pooled = jnp.mean(embedded, axis=1)
        hidden = jax.nn.relu(pooled @ params["hidden_w"] + params["hidden_b"])
        return hidden @ params["out_w"] + params["out_b"]


def main() -> None:
    rng = np.random.default_rng(0)
    params = {
        "embedding": jnp.asarray(rng.normal(size=(16, 8)).astype(np.float16)),
        "hidden_w": jnp.asarray(rng.normal(size=(8, 12)).astype(np.float16)),
        "hidden_b": jnp.asarray(rng.normal(size=(12,)).astype(np.float16)),
        "out_w": jnp.asarray(rng.normal(size=(12, 4)).astype(np.float16)),
        "out_b": jnp.asarray(rng.normal(size=(4,)).astype(np.float16)),
    }
    model = TinyClassifier()
    example_tokens = jnp.asarray([[1, 2, 3, 0]], dtype=jnp.int32)
    bundle_dir = Path("/private/tmp/cactus_jax_user_graph_minimal")

    result = build_jax_user_graph_bundle(
        params=params,
        output_dir=bundle_dir,
        model_id="tiny_jax_classifier",
        task="classification",
        specs=(
            JaxGraphSpec(
                name="classifier",
                fn=model,
                example_args=(example_tokens,),
                input_names=("token_ids",),
                output_names=("logits",),
            ),
        ),
    )

    loaded = load_jax_user_graph_bundle(result.output_dir)
    tokens = np.asarray([[4, 5, 6, 0]], dtype=np.int32)
    cactus_logits = loaded.execute("classifier", tokens)[0].numpy().astype(np.float32)
    jax_logits = np.asarray(model(params, jnp.asarray(tokens))).astype(np.float32)

    print("Bundle dir:", result.output_dir)
    print("Max diff:", float(np.max(np.abs(cactus_logits - jax_logits))))
    print("Logits:", cactus_logits)


if __name__ == "__main__":
    main()
