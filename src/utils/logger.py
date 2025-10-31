from loguru import logger
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logging(config: Config, *, override_file: Optional[str] = None):
    cfg = config.get().logging
    logger.remove()
    if cfg.console:
        logger.add(lambda msg: print(msg, end=""), level=cfg.level)
    log_file = override_file or cfg.file
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, level=cfg.level, format=cfg.format, rotation=cfg.rotation, retention=cfg.retention)
    return logger
