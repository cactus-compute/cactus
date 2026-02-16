from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .tensor_io import save_tensor_with_header


def convert_silero_vad_weights(model, output_dir, precision="FP16", args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    stft_basis = state_dict["_model.stft.forward_basis_buffer"]
    n_fft_bins, _, window_size = stft_basis.shape

    lstm_weight_ih = state_dict["_model.decoder.rnn.weight_ih"]
    lstm_hidden_size = lstm_weight_ih.shape[1]

    encoder_channels = []
    for i in range(4):
        key = f"_model.encoder.{i}.reparam_conv.weight"
        if key in state_dict:
            weight = state_dict[key]
            out_ch, in_ch, kernel = weight.shape
            encoder_channels.append((in_ch, out_ch, kernel))

    config = {
        "model_type": "silero_vad",
        "sampling_rate": 16000,
        "window_size": int(window_size),
        "n_fft_bins": int(n_fft_bins),
        "num_encoder_blocks": len(encoder_channels),
        "lstm_hidden_size": int(lstm_hidden_size),
        "model_variant": "default",
        "precision": precision,
    }

    save_tensor_with_header(
        stft_basis, output_dir / "stft_basis.weights", precision=precision
    )

    for i in range(config["num_encoder_blocks"]):
        save_tensor_with_header(
            state_dict[f"_model.encoder.{i}.reparam_conv.weight"],
            output_dir / f"encoder_block_{i}_conv_weight.weights",
            precision=precision,
        )
        save_tensor_with_header(
            state_dict[f"_model.encoder.{i}.reparam_conv.bias"],
            output_dir / f"encoder_block_{i}_conv_bias.weights",
            precision=precision,
        )

    lstm_weights = [
        ("_model.decoder.rnn.weight_ih", "lstm_weight_ih.weights"),
        ("_model.decoder.rnn.weight_hh", "lstm_weight_hh.weights"),
        ("_model.decoder.rnn.bias_ih", "lstm_bias_ih.weights"),
        ("_model.decoder.rnn.bias_hh", "lstm_bias_hh.weights"),
    ]
    for key, filename in lstm_weights:
        save_tensor_with_header(
            state_dict[key], output_dir / filename, precision="FP16"
        )

    save_tensor_with_header(
        state_dict["_model.decoder.decoder.2.weight"],
        output_dir / "output_conv_weight.weights",
        precision=precision,
    )
    save_tensor_with_header(
        state_dict["_model.decoder.decoder.2.bias"],
        output_dir / "output_conv_bias.weights",
        precision=precision,
    )

    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    return config
