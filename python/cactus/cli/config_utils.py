import json
import os
import tempfile
from pathlib import Path


class CactusConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".cactus"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        try:
            os.chmod(self.config_dir, 0o700)
        except OSError:
            pass

    def load_config(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def save_config(self, config):
        payload = json.dumps(config, indent=2)
        fd, tmp_path = tempfile.mkstemp(prefix="config.", suffix=".tmp", dir=str(self.config_dir))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.config_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_api_key(self):
        env_key = os.getenv("CACTUS_CLOUD_KEY") or os.getenv("CACTUS_CLOUD_API_KEY")
        if env_key:
            return env_key
        return self.load_config().get("api_key") or None

    def set_api_key(self, key):
        config = self.load_config()
        config["api_key"] = key
        self.save_config(config)

    def clear_api_key(self):
        config = self.load_config()
        config.pop("api_key", None)
        self.save_config(config)
