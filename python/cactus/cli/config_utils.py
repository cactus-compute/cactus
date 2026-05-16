import json
import os
from pathlib import Path


class CactusConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".cactus"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self.telemetry_cache_dir = Path.home() / "Library" / "Caches" / "cactus" / "telemetry"
        self.cloud_api_key_cache_file = self.telemetry_cache_dir / "cloud_api_key"

    def load_config(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def save_config(self, config):
        self.config_file.write_text(json.dumps(config, indent=2))

    def get_api_key(self):
        env_key = os.getenv("CACTUS_CLOUD_KEY")
        if not env_key:
            env_key = os.getenv("CACTUS_CLOUD_API_KEY")
        if env_key:
            return env_key
        config = self.load_config()
        config_key = config.get("api_key", "")
        if config_key:
            return config_key
        return self.load_cached_api_key()

    def cache_api_key(self, key):
        if not key:
            return
        self.telemetry_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_api_key_cache_file.write_text(key)

    def load_cached_api_key(self):
        if not self.cloud_api_key_cache_file.exists():
            return ""
        return self.cloud_api_key_cache_file.read_text().strip()

    def clear_cached_api_key(self):
        if self.cloud_api_key_cache_file.exists():
            self.cloud_api_key_cache_file.unlink()

    def set_api_key(self, key):
        config = self.load_config()
        config["api_key"] = key
        self.save_config(config)
        self.cache_api_key(key)

    def clear_api_key(self):
        config = self.load_config()
        config.pop("api_key", None)
        self.save_config(config)
        self.clear_cached_api_key()
