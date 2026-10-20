"""Locate external measurement sources without embedding historical task names."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def external_source(key):
    """Read a source path from the external registry or an explicit override.

    This registry locates existing evidence; it does not reinterpret a dataset,
    change a model or select a checkpoint. Missing records return a neutral
    prospective location so CLI planning remains available before setup.
    """
    override = os.environ.get('GIFT_SOURCE_' + key.upper())
    if override:
        value = override
    else:
        registry = Path(os.environ.get('GIFT_EXPERIMENT_SOURCE_REGISTRY',
                                      ROOT.parent / 'experiment_sources.json')).expanduser().resolve()
        if registry == ROOT or ROOT in registry.parents:
            raise ValueError('Experiment source registry must be outside GIFT')
        sources = json.loads(registry.read_text(encoding='utf-8')) if registry.is_file() else {}
        value = sources.get('sources', {}).get(key, {}).get('path', ROOT.parent / 'runs' / key)
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('External measurement sources must remain outside GIFT')
    return path
