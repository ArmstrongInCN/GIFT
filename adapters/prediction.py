"""Experiment-specific history layout around native external one-step models."""
from __future__ import annotations

import torch

from .models import load_checkpoint
from training.budgets import PREDICTION_EPOCHS


def _load(method, path, device):
    model, payload = load_checkpoint(method, path, str(device))
    epochs = PREDICTION_EPOCHS
    if payload.get("status") != "complete" or int(payload.get("terminal_epoch", -1)) != epochs:
        raise ValueError(f"{method}: a completed {epochs}-epoch terminal model is required")
    if payload.get("artifact_role") == "test_only" or payload.get("formal_configuration") is False:
        raise ValueError("A tiny training fixture cannot supply formal predictions")
    model.requires_grad_(False)
    return model, payload


def load_uno(path, device="cpu"):
    return _load("uno", path, device)


def load_unet(path, device="cpu"):
    return _load("unet", path, device)


def _rollout(model, history, steps, channels_last):
    if history.ndim != 4 or history.shape[1:] != (46, 64, 64) or steps < 1:
        raise ValueError("Expected [batch,46,64,64] context and positive future steps")
    window = history.permute(0, 2, 3, 1).contiguous() if channels_last else history
    axis = -1 if channels_last else 1
    predictions = []
    for _ in range(steps):
        raw = model(window)
        expected = (len(history), 64, 64, 1) if channels_last else (len(history), 1, 64, 64)
        if tuple(raw.shape) != expected:
            raise ValueError("External model prediction shape differs")
        predictions.append(raw[..., 0] if channels_last else raw[:, 0])
        old = window[..., 1:] if channels_last else window[:, 1:]
        window = torch.cat((old, raw), dim=axis)
    return torch.stack(predictions, dim=1)


def uno_rollout(model, history, steps=150):
    return _rollout(model, history, steps, True)


def unet_rollout(model, history, steps=150):
    return _rollout(model, history, steps, False)
