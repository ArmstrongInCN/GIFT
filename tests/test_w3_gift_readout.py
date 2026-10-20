"""Synthetic coefficient readout fixtures; no scientific weights or training."""
import h5py
import numpy as np
import pytest
import torch

from experiments.formal.w3_equation_identification import run


@pytest.mark.parametrize("recorded_hash", [None, "absent", "mismatch"])
def test_gift_readout_accepts_optional_data_hash_and_reads_actual_fields(tmp_path, monkeypatch, recorded_hash):
    """Coefficient projection must use real fixture fields and check an optional recorded observation hash.

    An analytic stand-in supplies known actions so the projection/sign conventions
    can be tested independently of trained generator accuracy.
    """
    import gift
    from even_full_spectrum_ns import EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver

    dataset = tmp_path / "synthetic_observations.h5"
    values = np.random.default_rng(14).normal(size=(1, 11, 64, 64)).astype(np.float32)
    with h5py.File(dataset, "x") as handle:
        group = handle.create_group("training")
        group.create_dataset("trajectory_index", data=np.asarray([0], dtype=np.int64))
        group.create_dataset("time", data=5. + np.arange(11) * .1)
        group.create_dataset("vorticity", data=values)

    class AnalyticFixtureGenerator(torch.nn.Module):
        """Provide analytic Fourier actions with known coefficients solely for readout verification."""
        def __init__(self, cutoff, rank):
            """Create a CPU periodic reference solver and integer squared-wave-number grid."""
            super().__init__()
            self.solver = EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(
                grid=64, padding_grid=99, viscosity=.01, forcing_amplitude=1.,
                forcing_wavenumber=4, dt=.005), device="cpu")
            frequencies = torch.fft.fftfreq(64, d=1. / 64)
            ky, kx = torch.meshgrid(frequencies, frequencies, indexing="ij")
            self.k2 = kx.square() + ky.square()

        def components_hat(self, state):
            """Return unit forcing/transport and nu=0.01 linear action in native Fourier coordinates."""
            coefficients = torch.fft.fft2(state)
            return {"constant": self.solver.forcing_hat.expand_as(coefficients),
                    "quadratic": -self.solver.transport_hat(coefficients),
                    "linear": -.01 * self.k2 * coefficients}

    monkeypatch.setattr(gift, "GIFTGenerator", AnalyticFixtureGenerator)
    training_data = {} if recorded_hash == "absent" else {"sha256": recorded_hash}
    if recorded_hash == "mismatch":
        training_data["sha256"] = "0" * 64
    checkpoint = tmp_path / "synthetic_weights.pt"
    torch.save({"format_version": 3, "artifact_role": "trained_generator",
                "training_data": training_data, "model_configuration": {"cutoff": 21, "rank": 8},
                "model_state_dict": {}}, checkpoint)
    if recorded_hash == "mismatch":
        with pytest.raises(ValueError, match="checkpoint/data binding"):
            run.gift_readout(run.ROOT, dataset, checkpoint, "cpu")
    else:
        monkeypatch.setattr(run, "digest", lambda _: pytest.fail("optional hash caused a whole-file scan"))
        measured = run.gift_readout(run.ROOT, dataset, checkpoint, "cpu")
        assert measured["nu"] == pytest.approx(.01, abs=1e-8)
        assert measured["beta"] == pytest.approx(1., abs=1e-7)
        assert measured["gamma"] == pytest.approx(1., abs=1e-7)
