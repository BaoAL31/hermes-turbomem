from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_DATA_DIR = Path.home() / ".hermes" / "turbomem"
CONFIG_FILENAME = "config.yaml"


@dataclass
class TurbomemConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    auto_index_on_first_use: bool = False
    embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    bit_width: int = 4
    default_recall_limit: int = 8

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.tvim"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def config_path(self) -> Path:
        return self.data_dir / CONFIG_FILENAME


def load_config() -> TurbomemConfig:
    data_dir = Path(os.environ.get("TURBOMEM_DATA_DIR", DEFAULT_DATA_DIR))
    config_path = data_dir / CONFIG_FILENAME
    cfg = TurbomemConfig(data_dir=data_dir)

    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if "data_dir" in raw:
            cfg.data_dir = Path(raw["data_dir"]).expanduser()
        cfg.auto_index_on_first_use = bool(raw.get("auto_index_on_first_use", False))
        cfg.embedding_model = str(raw.get("embedding_model", cfg.embedding_model))
        cfg.bit_width = int(raw.get("bit_width", cfg.bit_width))
        cfg.default_recall_limit = int(raw.get("default_recall_limit", cfg.default_recall_limit))

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
