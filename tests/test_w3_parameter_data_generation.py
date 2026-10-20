"""Bounded CPU contracts for the clean fifty-equation W3 observation generator.

These tests never train a model or generate the full formal trajectory cohort.
Metadata-only fixtures are explicitly mock observations, not scientific data.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts import generate_w3_parameter_data as generator
from src.even_full_spectrum_ns import (
    EvenFullSpectrumConfiguration, EvenFullSpectrumNSSolver, build_initial_hat,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    """Keep tiny CPU solver tests bounded and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class LegacyRemainderSolver(EvenFullSpectrumNSSolver):
    """Retain the pre-adaptation unit-beta remainder for numerical comparison."""
    def remainder(self, value):
        return -self.transport_hat(value) + self.forcing_hat

    def _remainder(self, value):
        return -self._transport_hat(value) + self.forcing_hat


def test_default_beta_preserves_original_step_path_exactly():
    """Default and explicit beta=1 agree bitwise with the original RHS path."""
    initial = generator.initial_conditions()["splits"]["training"]["parameters"]
    state = build_initial_hat(initial)
    for configuration in (EvenFullSpectrumConfiguration(),
                          EvenFullSpectrumConfiguration(advection_coefficient=1.)):
        old = LegacyRemainderSolver(configuration)
        new = EvenFullSpectrumNSSolver(configuration)
        a, b = state.clone(), state.clone()
        assert torch.equal(new.remainder(state), old.remainder(state))
        for _ in range(8):
            a, b = old.advance(a), new.advance(b)
            assert torch.equal(a, b)


def test_actual_coefficients_and_all_four_stages(monkeypatch):
    """Beta multiplies unit transport in every stage; nu and gamma remain actual."""
    truth = {"nu": .012, "beta": .81, "gamma": 1.17}
    solver = generator.make_solver(truth)
    state = build_initial_hat(generator.initial_conditions()["splits"]["training"]["parameters"])
    expected = -truth["beta"] * solver.transport_hat(state) + solver.forcing_hat
    assert torch.equal(solver.remainder(state), expected)
    torch.testing.assert_close(solver.linear, -truth["nu"] * solver.k2, rtol=0, atol=0)
    unit = generator.make_solver({"nu": .01, "beta": 1., "gamma": 1.})
    # Forcing is formed by the same float32 physical-grid operation, so compare
    # to the exact declared construction rather than scalar multiplication of
    # an already rounded Fourier spectrum.
    coordinate = 2*np.pi*torch.arange(64,dtype=torch.float32)/64
    y, _x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    forcing = -truth["gamma"]*4*torch.cos(4*y)
    assert torch.equal(solver.forcing_hat, torch.fft.fft2(forcing))
    calls = []
    original = solver._remainder

    def tracked(value):
        result = original(value)
        assert torch.equal(result, -truth["beta"]*solver._transport_hat(value)+solver.forcing_hat)
        calls.append(value.clone())
        return result

    monkeypatch.setattr(solver, "_remainder", tracked)
    result = solver.advance(state)
    assert len(calls) == 4
    assert torch.isfinite(result).all()
    assert not torch.equal(result, unit.advance(state))
    assert solver.metadata["advection_coefficient"] == truth["beta"]
    with pytest.raises(ValueError, match="finite"):
        EvenFullSpectrumNSSolver(EvenFullSpectrumConfiguration(advection_coefficient=float("nan")))


def test_initial_population_and_observation_schedule():
    """Use original ID0/validation20 only, with all 501 exact solver-save nodes."""
    initial = generator.initial_conditions()
    pools = generator.seed_parameters()
    np.testing.assert_array_equal(initial["splits"]["training"]["ids"], [0])
    np.testing.assert_array_equal(initial["splits"]["training"]["parameters"], pools["training"][1][:1])
    np.testing.assert_array_equal(initial["splits"]["validation"]["ids"], np.arange(50,70))
    np.testing.assert_array_equal(initial["splits"]["validation"]["parameters"], pools["validation"][1])
    np.testing.assert_array_equal(generator.SAVED_STEPS, np.arange(0,2001,4))
    np.testing.assert_array_equal(generator.TIMES, np.arange(501)*.02)


def test_short_real_integration_uses_coefficient_and_same_initial_field():
    """Two actual eight-step integrations differ after t0, not in their initial state."""
    initial = generator.initial_conditions()["splits"]["training"]["parameters"]
    a, meta_a = generator.integrate_batch(initial, {"nu": .01, "beta": .75, "gamma": 1.}, saved_steps=np.array([0,4,8]))
    b, meta_b = generator.integrate_batch(initial, {"nu": .01, "beta": 1.25, "gamma": 1.}, saved_steps=np.array([0,4,8]))
    assert a.shape == (1,3,64,64) and a.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(a[:,0],b[:,0])
    assert not np.array_equal(a[:,1:],b[:,1:])
    assert meta_a["advection_coefficient"] == .75
    assert meta_b["advection_coefficient"] == 1.25
    with pytest.raises(ValueError, match="save indices"):
        generator.integrate_batch(initial,{"nu": .01,"beta":1.,"gamma":1.},saved_steps=np.array([1,4]))


def test_csv_hash_prevents_new_draws_and_scope_escape(tmp_path, monkeypatch):
    """Reject a different parameter design, invalid cases and project-local output."""
    path = tmp_path/"parameters.csv"
    path.write_text("case_id,nu,beta,gamma\nparameter_000,.01,1,1\n",encoding="utf-8")
    with pytest.raises(ValueError,match="locked fifty"):
        generator.parameter_cases(path)
    monkeypatch.setattr(generator,"parameter_cases",lambda path:[{"case_id":"parameter_000","truth":{"nu":.01,"beta":1.,"gamma":1.}}])
    with pytest.raises(ValueError,match="outside"):
        generator.main(["--root",str(generator.ROOT/"generated_data"),"--parameter-csv",str(path)])
    with pytest.raises(ValueError,match="fixed design"):
        generator.main(["--root",str(tmp_path/"run"),"--parameter-csv",str(path),"--case-id","parameter_001"])
    output=tmp_path/"planned"
    assert generator.main(["--root",str(output),"--parameter-csv",str(path)])==0
    assert not output.exists()


def test_unit_basis_rank_and_oracle_error():
    """Recover all three known coefficients and reject an unidentifiable basis."""
    rng=np.random.default_rng(1)
    x=rng.normal(size=(64,3))
    truth={"nu":.009,"beta":1.1,"gamma":.85}
    y=x@np.asarray(list(truth.values()))
    result=generator.least_squares(x,y,truth)
    np.testing.assert_allclose(result["coefficients"],list(truth.values()),atol=1e-14,rtol=0)
    assert result["rank"]==3 and max(result["ape_percent"])<1e-10
    with pytest.raises(ValueError,match="rank deficient"):
        generator.least_squares(np.column_stack([x[:,0],x[:,0],x[:,2]]),y,truth)


def test_resume_refuses_foreign_case_and_changed_lock(tmp_path):
    """A foreign equation lock cannot be silently reused or overwritten."""
    case={"case_id":"parameter_000","truth":{"nu":.01,"beta":1.,"gamma":1.}}
    folder=tmp_path/case["case_id"]
    folder.mkdir()
    generator.write_new(folder/"DATA_LOCK.json",generator.data_lock(case,"first_initial_hash"))
    before=(folder/"DATA_LOCK.json").read_bytes()
    with pytest.raises(ValueError,match="configuration differs"):
        generator.generate_case(tmp_path,case,"different_initial_hash",resume=True)
    assert (folder/"DATA_LOCK.json").read_bytes()==before
    with pytest.raises(FileExistsError):
        generator.generate_case(tmp_path,case,"first_initial_hash",resume=False)


def test_audit_rejects_wrong_population_before_reading_or_fitting(tmp_path):
    """The finite/population gate cannot accept duplicated fifty-training containers."""
    case={"case_id":"parameter_000","truth":{"nu":.01,"beta":1.,"gamma":1.}}
    lock=generator.data_lock(case,"test_only")
    path=tmp_path/"MOCK_WRONG_POPULATION.h5"
    with h5py.File(path,"x") as h:
        h.attrs.update(case_id=case["case_id"],truth_json=json.dumps(case["truth"]),role="MOCK_NOT_SCIENTIFIC_DATA")
        training=h.create_group("training")
        training.create_dataset("vorticity",shape=(50,501,64,64),dtype="f4",chunks=(1,1,64,64),fillvalue=np.nan)
    with pytest.raises(ValueError,match="population"):
        generator.audit_case(path,lock)


def test_case_lease_and_evidence_are_non_destructive(tmp_path):
    """OS lease release keeps evidence and finite JSON creation never overwrites."""
    lease=tmp_path/"lease.lock"
    with generator.case_lease(lease):
        with pytest.raises(OSError):
            with generator.case_lease(lease):
                pytest.fail("a second lease must not succeed")
    with generator.case_lease(lease):
        assert lease.is_file()
    path=tmp_path/"record.json"
    generator.write_new(path,{"status":"test"})
    with pytest.raises(FileExistsError):
        generator.write_new(path,{"status":"replaced"})
    assert json.loads(path.read_text())=={"status":"test"}
