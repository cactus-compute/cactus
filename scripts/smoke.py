import array
import json
import math
import sys
import tempfile
from pathlib import Path

from PIL import Image

from cactus import Cactus, Mirage, Whistle, tool
from cactus.agent import fetch, runtime


@tool
def set_thermostat(temperature: int):
    """Set the thermostat temperature."""
    return {"temperature": temperature}


def check_text(agent):
    response = agent.complete("set the thermostat to 21 degrees")
    assert response["type"] == "call", response
    call = response["function_calls"][0]
    assert call["name"] == "set_thermostat", response
    assert call["arguments"]["temperature"] == 21, response
    embedding = agent.embed("warm room")
    assert embedding and all(math.isfinite(value) for value in embedding)


agent = Cactus(tools=[set_thermostat], auto_date=False, stateless=True)
check_text(agent)
silence = array.array("f", [0]) * 16000
speech = Whistle()
assert "text" in speech.transcribe(silence, language="en")
assert list(speech.stream([silence], language="en"))
assert speech.embed(silence)
vision = Mirage()
pixels = bytes([90, 100, 110]) * (32 * 32)
result = vision.describe(pixels, width=32, height=32, sections=["objects"], max_tokens=8)
assert isinstance(result["objects"], list), result
with tempfile.TemporaryDirectory() as directory:
    image = Path(directory) / "image.png"
    Image.frombytes("RGB", (32, 32), pixels).save(image)
    decoded = vision.describe(image, sections=["objects"], max_tokens=8)
    assert decoded == result, decoded
check_text(agent)
agent.reset()
assert "audio_text" in agent.complete(audio=silence, language="en", max_new_tokens=16)
with Cactus(weights=fetch.fetch_weights(), tools=[set_thermostat], auto_date=False) as custom:
    check_text(custom)
    custom.reset()
    assert "audio_text" in custom.complete(audio=silence, language="en", max_new_tokens=16)
    custom.reset()
    check_text(custom)
    assert "audio_text" in custom.complete(audio=silence, language="en", max_new_tokens=16)
assert runtime._lib().cactus_models() == 7
assert "jax" not in sys.modules
print(json.dumps({"platform": fetch._platform_tag(), "engine": fetch.ENGINE_VERSION,
                  "revision": fetch.ENGINE_REVISION, "text": True, "speech": True, "vision": True}))
