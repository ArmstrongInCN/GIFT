"""Small CPU file fixture for seven-slot refresh, resume and cache identity.

Model loading and prediction are stubs. Uniform synthetic fields permit exact
analytic metric files; no network or physical dataset is used by these tests.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.additional import w1_anie_coordinate_refresh as evaluation


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluation.shared._json_ready(value), allow_nan=False), encoding="utf-8")


def fields(path, *, source=False):
    """HDF fill values allocate no180-by76 field array on disk."""
    with h5py.File(path, "x") as f:
        f.attrs.update(status="complete", trajectory_id_scheme="canonical", complete=True)
        group = f.create_group("test") if source else f
        group.create_dataset("trajectory_index", data=np.arange(100000,100180))
        group.create_dataset("time", data=evaluation.native.QUERY_TIMES)
        group.create_dataset("absolute_time", data=5+evaluation.native.QUERY_TIMES)
        group.create_dataset("vorticity", shape=(180,76,64,64), dtype="f4", fillvalue=1.)


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Every file belongs to a synthetic isolated run; all model calls are replaced."""
    root, old = tmp_path/"refresh",tmp_path/"previous"
    previous = old/"w1_evaluation/confirmation"
    (previous/"raw").mkdir(parents=True)
    (previous/"metrics").mkdir()
    source = tmp_path/"source.h5"
    fields(source,source=True)
    fields(previous/"raw/truth.h5")
    checkpoint = root/"training/anie_ns_seed7/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint-not-a-torch-model")
    plan = dict(old_root=str(old), source_profile="aligned_ns_single_v1",device="cpu",evaluation_batch=4,
                job_events_file=str(root/"JOB_EVENTS.jsonl"))
    lock = dict(profile="single_native76_n300", fixture=True)
    dump(root/"RUN_PLAN.json",plan)
    dump(root/"PROTOCOL_LOCK.json",lock)
    dump(root/"FORMAL.json",dict(status="complete",checkpoint=str(checkpoint)))
    seeds = dict(anie_ns=7,fno2d=0,fno3d=0,GIFT=20260820,uno=0,unet=0,**{"GIFT-N50":20260820})
    records = {}
    old_bytes = {}
    for method,seed in seeds.items():
        raw,metric = previous/"raw"/(method+".h5"),previous/"metrics"/(method+".h5")
        raw.write_bytes(("immutable fixture raw "+method).encode())
        metric.write_bytes(("immutable fixture metrics "+method).encode())
        old_bytes[str(raw)],old_bytes[str(metric)] = raw.read_bytes(),metric.read_bytes()
        records[method] = dict(method=method,seed=seed,status="complete",checkpoint="old-fixture",
            checkpoint_sha256="c"*64,raw_prediction_file=str(raw.relative_to(previous)),
            raw_prediction_sha256=evaluation.shared.file_sha256(raw),metric_file=str(metric.relative_to(previous)),
            metric_sha256=evaluation.shared.file_sha256(metric),summary=dict(population=180,failed_count=0,primary_metric=.1),
            per_trajectory=dict(trajectory_id=list(range(100000,100180))),new_inference_performed=True)
    report = dict(schema="gift.w1-extended-native.evaluation.v1",status="numerical_complete",cohort="confirmation",
                  population=180,trajectory_ids=list(range(100000,100180)),query_times=evaluation.native.QUERY_TIMES.tolist(),
                  absolute_times=(5+evaluation.native.QUERY_TIMES).tolist(),time_points=76,future_steps=75,dt=.04,
                  method_seeds=seeds,methods=records,extension_lock=dict(historical=True))
    dump(previous/"numeric_report.json",report)
    dump(previous/"evaluation_plan.json",dict(data_file=str(source),group="test"))
    spec = dict(method="anie_ns",history=1,source_profile="aligned_ns_single_v1",condition_mode="native",
                query_times=evaluation.native.QUERY_TIMES.tolist(),future_steps=75,observation_dt=.04)
    payload = dict(model_spec=spec,epoch=28,config=dict(seed=7),data_binding=dict(fixture=True),
                   source_provenance=dict(upstream=dict(source_profile="aligned_ns_single_v1")))
    monkeypatch.setattr(evaluation.native,"checkpoint_metadata",lambda *args:payload)
    monkeypatch.setattr(evaluation.native,"load_checkpoint",lambda *args:pytest.fail("No model loading is allowed"))
    return dict(root=root,previous=previous,checkpoint=checkpoint,payload=payload,old_bytes=old_bytes)


def predicted_file(path, checkpoint, payload, *, completed=180, previous=None):
    """Write a uniform numerical fixture, optionally copying its committed prefix."""
    with h5py.File(path,"x") as f:
        f.attrs.update(status="complete" if completed==180 else "incomplete",checkpoint=str(checkpoint),
                       model_spec_json=json.dumps(payload["model_spec"]),completed_trajectories=completed,
                       inference_cost_json=json.dumps(dict(inference_seconds=1.,fixture=True)),
                       inference_input="omega(t5) only",time_points=76,future_steps=75)
        f.create_dataset("trajectory_index",data=np.arange(100000,100180))
        f.create_dataset("time",data=evaluation.native.QUERY_TIMES)
        f.create_dataset("absolute_time",data=5+evaluation.native.QUERY_TIMES)
        data = f.create_dataset("prediction",shape=(180,76,64,64),dtype="f4",fillvalue=1.,
                                chunks=(1,1,64,64),compression="lzf")
        f.create_dataset("solver_failed",data=np.zeros(180,bool))
        f.create_dataset("failure_reason",data=np.asarray([""]*180,dtype=h5py.string_dtype()))
        if previous:
            with h5py.File(previous,"r") as prior:
                assert prior.attrs["completed_trajectories"] == 2
                data[:2] = prior["prediction"][:2]
        elif completed==2:
            data[:2] = .5


def analytic_uniform_score(raw_path, truth, metric_path):
    """Correct exact metrics for constant synthetic frames, without large reductions."""
    assert truth.shape == (180,76,64,64)
    with h5py.File(raw_path,"r") as raw:
        relative = np.abs(raw["prediction"][:,:,0,0].astype(np.float64)-1.)
    per = dict(trajectory_id=np.arange(100000,100180),future_mean_relative_l2=relative[:,1:].mean(1),
               t8_relative_l2=relative[:,-1],future_mse=np.square(relative[:,1:]).mean(1),
               future_path_relative_l2=np.sqrt(np.square(relative[:,1:]).mean(1)),
               t0_relative_l2=relative[:,0],all_points_mse_diagnostic=np.square(relative).mean(1))
    metrics = {name:evaluation.shared._summary(values) for name,values in per.items() if name!="trajectory_id"}
    summary = dict(status="complete",population=180,failed_count=0,failed_ids=[],failure_rate=0.,
                   primary_metric_name="future_mean_per_frame_relative_l2",primary_metric=metrics["future_mean_relative_l2"]["mean"],
                   selection_metric=metrics["future_mean_relative_l2"]["mean"],metrics=metrics,
                   mean_relative_l2_by_time=relative.mean(0).tolist(),t0_in_primary_score=False,time_points=76,future_steps=75,
                   query_times=evaluation.native.QUERY_TIMES.tolist(),absolute_times=(5+evaluation.native.QUERY_TIMES).tolist(),dt=.04)
    with h5py.File(metric_path,"x") as f:
        f.attrs["status"] = "complete"
        f.create_dataset("time",data=evaluation.native.QUERY_TIMES)
        f.create_dataset("absolute_time",data=5+evaluation.native.QUERY_TIMES)
        f.create_dataset("relative_l2_by_time",data=relative)
        f.create_dataset("finite_by_time",data=np.ones((180,76),bool))
        f.create_dataset("failed",data=np.zeros(180,bool))
        f.create_dataset("solver_failed",data=np.zeros(180,bool))
        f.create_dataset("first_nonfinite_future_index",data=np.full(180,-1,dtype=np.int64))
        for key,values in per.items(): f.create_dataset("per_trajectory/"+key,data=values)
    return dict(summary=summary,per_trajectory=per)


def test_partial_raw_resume_reuses_six_slots_and_rejects_changed_cache(study,monkeypatch):
    """One interrupted prefix resumes; marker repair requires no second prediction."""
    calls = []
    def predictor(checkpoint,payload,window,path,**kwargs):
        calls.append(kwargs.get("previous_raw"))
        if len(calls)==1:
            predicted_file(path,checkpoint,payload,completed=2)
            raise RuntimeError("synthetic interrupted numerical file write")
        assert calls[-1].name == "anie_ns.interrupted_000.h5"
        predicted_file(path,checkpoint,payload,previous=calls[-1])
        return dict(inference_seconds=1.,fixture=True)
    monkeypatch.setattr(evaluation.native,"predict_checkpoint",predictor)
    monkeypatch.setattr(evaluation.shared,"score_raw_predictions",analytic_uniform_score)
    with pytest.raises(RuntimeError,match="synthetic interrupted"):
        evaluation.evaluate(study["root"],"confirmation")
    report = evaluation.evaluate(study["root"],"confirmation",resume=True)
    output = study["root"]/"w1_evaluation/confirmation"
    assert evaluation.complete_evaluation(output)
    assert len(report["methods"])==7 and report["methods"]["anie_ns"]["summary"]["population"]==180
    assert report["methods"]["anie_ns"]["summary"]["primary_metric"]==pytest.approx(1/180)
    assert "extension_lock" not in report and report["prior_extension_lock"]==dict(historical=True)
    assert report["base_lock"]==evaluation.read(study["root"]/"PROTOCOL_LOCK.json")
    old_report = evaluation.read(study["previous"]/"numeric_report.json")
    for method,original in old_report["methods"].items():
        if method=="anie_ns": continue
        copied = report["methods"][method]
        assert copied["summary"]==original["summary"] and not copied["new_inference_performed"]
        for key,sha in (("raw_prediction_file","raw_prediction_sha256"),("metric_file","metric_sha256")):
            assert evaluation.shared.file_sha256(output/copied[key])==original[sha]
    for name,value in study["old_bytes"].items(): assert Path(name).read_bytes()==value
    marker = output/"EVALUATION_COMPLETE.json"
    marker.write_text('{"status":',encoding="utf-8")
    assert not evaluation.complete_evaluation(output)
    repaired = evaluation.evaluate(study["root"],"confirmation",resume=True)
    assert len(calls)==2 and evaluation.complete_evaluation(output)
    assert evaluation.shared._json_ready(repaired)==evaluation.shared._json_ready(report)
    assert (output/"EVALUATION_COMPLETE.interrupted_000.json").is_file()
    study["checkpoint"].write_bytes(b"changed-only-this-synthetic-checkpoint")
    assert not evaluation.complete_evaluation(output)
    with pytest.raises(ValueError,match="same model, protocol"):
        evaluation.evaluate(study["root"],"confirmation",resume=True)


@pytest.mark.parametrize("kind",["numerical","resource","infrastructure"])
def test_scientific_failure_retains180_and_infrastructure_cannot_complete(study,monkeypatch,kind):
    """Failure identity remains bound to its actual file, with no fake prediction."""
    failure = study["root"]/"FAILURE.json"
    dump(failure,dict(failure_kind=kind,reason="synthetic failure"))
    dump(study["root"]/"FORMAL.json",dict(status="failed",failure=str(failure)))
    monkeypatch.setattr(evaluation.native,"predict_checkpoint",lambda *args,**kwargs:pytest.fail("Failure cannot predict"))
    if kind=="infrastructure":
        with pytest.raises(ValueError,match="infrastructure interruption"):
            evaluation.evaluate(study["root"],"confirmation")
        assert not (study["root"]/"w1_evaluation/confirmation/EVALUATION_COMPLETE.json").exists()
        return
    report = evaluation.evaluate(study["root"],"confirmation")
    record = report["methods"]["anie_ns"]
    assert len(report["methods"])==7 and record["summary"]["failed_count"]==180
    assert record["summary"]["failed_ids"]==list(range(100000,100180))
    assert record["summary"]["primary_metric"] is None and not record["summary"]["metrics"]
    assert "raw_prediction_file" not in record
    output = study["root"]/"w1_evaluation/confirmation"
    assert evaluation.complete_evaluation(output)
    dump(failure,dict(failure_kind=kind,reason="changed failure identity"))
    assert not evaluation.complete_evaluation(output)
