"""CPU regressions for all-rejected GIFT correction batches and preserved failure scores.

Real RK4/projection/correction code runs on fixed P21/Q21 impulses. The FFT
spy rejects empty inputs while forwarding every nonempty tensor to real torch
FFT, so the old unconditional accepted-state composition fails this test.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.formal._shared import gift_runtime as runtime
import training.w1_v2_gift as training


def _fixtures():
    """Construct two native-gate outliers and a stable sinusoid on the actual N64 grid."""
    mask = runtime.square_selected_mask(64, cutoff=21, device=torch.device("cpu"))
    impulse = torch.zeros((64, 64), dtype=torch.float32)
    impulse[0, 0] = 1000.
    spectrum = torch.fft.fft2(impulse)
    p21 = torch.fft.ifft2(spectrum*mask).real.numpy()
    q21 = torch.fft.ifft2(spectrum*~mask).real.numpy()
    coordinate = np.arange(64, dtype=np.float32)/64
    stable = np.repeat(np.sin(2*np.pi*3*coordinate)[None], 64, axis=0)
    return np.stack((p21, q21, stable)).astype(np.float32)


def _zero_rhs(state):
    """Zero RHS isolates the existing local-correction gate and track projection."""
    return torch.zeros_like(state)


def _policies():
    """Use a one-pixel test cap to reliably exercise native failures; production defaults stay unchanged."""
    return (runtime.LocalCorrectionPolicy(maximum_flagged_fraction=1./4096),
            runtime.LocalCorrectionPolicy(amplitude_threshold=12., maximum_flagged_fraction=1./4096))


def _forbid_empty_fft(monkeypatch):
    """Assert that rejection handling never submits [0,N,N] to FFT; nonempty numerics remain real."""
    original = torch.fft.fft2
    shapes = []
    def checked(value, *args, **kwargs):
        shapes.append(tuple(value.shape))
        assert value.numel() > 0, "empty accepted batch entered FFT"
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.fft, "fft2", checked)
    return shapes


@pytest.mark.parametrize("diagnostics", [False, True])
def test_all_gate_rejections_skip_fft_and_keep_both_failure_ids(monkeypatch, diagnostics):
    """All-rejected batches retain native reasons, gate counts and NaN futures without empty FFT."""
    torch.set_num_threads(1)
    initial = _fixtures()[:2]
    p21, q21 = _policies()
    shapes = _forbid_empty_fft(monkeypatch)
    result = runtime.rollout_gift(low=_zero_rhs, branch=None, initial_state=initial,
                                 trajectory_ids=np.array([101, 202]), duration=.08,
                                 save_interval=.04, step=.02, device="cpu", batch_size=2,
                                 correction_policy=p21, q21_correction_policy=q21, diagnostics=diagnostics)
    assert shapes and all(shape[0] > 0 for shape in shapes)
    assert result.prediction.shape == (2, 3, 64, 64)
    np.testing.assert_array_equal(result.prediction[:, 0], initial)
    assert np.isnan(result.prediction[:, 1:]).all()
    assert result.failure_step.tolist() == [1, 1]
    assert result.failure_reason == ("correction_gate_p21", "correction_gate_q21")
    assert result.runtime["low_rhs_evaluations_by_trajectory"] == [4, 4]
    assert result.correction["gate_failed_trajectory_ids"] == [101, 202]
    assert result.correction["gate_failed_by_step_trajectory"] == [[True, True], [False, False], [False, False], [False, False]]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rejected_subbatches_do_not_change_nonempty_trajectory_numerics(monkeypatch, batch_size):
    """A stable nonempty path is bitwise identical alone or alongside failed batches of varying sizes."""
    torch.set_num_threads(1)
    initial = _fixtures()
    p21, q21 = _policies()
    arguments = dict(low=_zero_rhs, branch=None, duration=.08, save_interval=.04,
                     step=.02, device="cpu", correction_policy=p21, q21_correction_policy=q21)
    reference = runtime.rollout_gift(initial_state=initial[2:], trajectory_ids=np.array([303]),
                                    batch_size=1, **arguments)
    shapes = _forbid_empty_fft(monkeypatch)
    result = runtime.rollout_gift(initial_state=initial, trajectory_ids=np.array([101, 202, 303]),
                                 batch_size=batch_size, **arguments)
    assert all(shape[0] > 0 for shape in shapes)
    np.testing.assert_array_equal(result.prediction[2:], reference.prediction)
    np.testing.assert_array_equal(result.finite_by_time[2:], reference.finite_by_time)
    assert result.failure_step.tolist() == [1, 1, -1]
    assert result.failure_reason == ("correction_gate_p21", "correction_gate_q21", None)
    assert result.runtime["low_rhs_evaluations_by_trajectory"] == [4, 4, 16]
    assert np.isnan(result.prediction[:2, 1:]).all()


def test_all_validation_failures_keep_population_and_cannot_select_best(monkeypatch):
    """Twenty actual native-gate failures stay in future75 scoring; no finite subset becomes a selected model."""
    torch.set_num_threads(1)
    initial = np.repeat(_fixtures()[:1], 20, axis=0)
    ids = np.arange(1000, 1020).tolist()
    bank = SimpleNamespace(validation=np.repeat(initial[:, None], 76, axis=1),
                           binding={"validation_ids": ids, "query_times": (np.arange(76)*.04).tolist()})
    p21, q21 = _policies()
    shapes = _forbid_empty_fft(monkeypatch)
    def checked_rollout(**kwargs):
        return runtime.rollout_gift(**kwargs, correction_policy=p21, q21_correction_policy=q21)
    monkeypatch.setattr(training, "rollout_gift", checked_rollout)
    result = training._validation(_zero_rhs, None, bank, torch.device("cpu"), training.GIFTTrainingConfig())
    assert all(shape[0] > 0 for shape in shapes)
    assert result["failed_trajectory_count"] == 20
    assert result["failed_trajectory_ids"] == ids
    assert result["val_metric"] is None and result["t8_relative_l2"] is None
    assert result["future75_physical_mse"] is None
    assert len(result["relative_l2"]) == 20
    assert all(len(row) == 76 and all(value is None for value in row) for row in result["relative_l2"])
    model = torch.nn.Linear(1, 1)
    assert training._select(None, result, 20, model) is None
    existing = training._select(None, {"val_metric": .5}, 10, model)
    assert training._select(existing, result, 20, model) is existing
