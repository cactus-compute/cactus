![Cactus Compute - Sub-50mb Super Intelligence For Consumer Devices](assets/banner.svg)

Run text, speech and vision models locally with one Python package. Cactus provides tool calling, structured extraction, transcription, image understanding and model fine-tuning.

## Install

```sh
pip install -U cactus-compute
```

Python 3.9 or later. The package downloads and caches the engine and requested model weights on first use. Inference runs locally and does not require JAX.

## Agents

[Needle 3](https://cactuscompute.com/needle) calls tools, extracts structured data and embeds text.

```python
import cactus

@cactus.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 27, "sky": "clear"}

with cactus.Cactus(tools=[get_weather]) as agent:
    print(agent.run("What's the weather in Lagos?")["results"])
```

`run()` executes your functions and returns their results. Use `complete()` when you want to inspect the proposed calls before executing them.

## Voice

[Whistle](https://cactuscompute.com/blog/whistle) transcribes speech in seven languages, with word timestamps and speech embeddings.

```python
import cactus

print(cactus.transcribe("clip.wav")["text"])
```

Use a 16 kHz WAV file. Install `cactus-compute[mic]` for other sample rates or microphone capture.

## Vision

[Mirage](https://huggingface.co/Cactus-Compute/mirage) describes images and extracts structured data. Install `cactus-compute[vision]` for image files.

```python
import cactus

print(cactus.Mirage().describe("photo.png"))
```

## Documentation

- [Python API and CLI](https://cactuscompute.com/blog/needle-python-docs)
- [Runtime binaries and supported platforms](https://huggingface.co/Cactus-Compute/cactus)
- [Fine-tune Needle](https://cactuscompute.com/blog/finetuning-needle) or [Whistle](https://cactuscompute.com/blog/finetuning-whistle) with `cactus-compute[train]`
- [Reference for coding assistants](llms.txt)

Migrating from `cactus-needle`? Use `import cactus` and the `cactus` command. Needle 2 remains on the [legacy branch](https://github.com/cactus-compute/cactus/tree/legacy) with `cactus-compute<4`. The earlier general inference SDK is now [cactus-engine](https://pypi.org/project/cactus-engine/).

Anonymous Python usage counts can be disabled with `CACTUS_TELEMETRY=0`. See [LICENSE](LICENSE) for the Apache 2.0 license.
