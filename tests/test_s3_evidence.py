"""S3 evidence must preserve population failures and the original W1 panel."""
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from experiments.formal.w1_recursive_prediction.run import gift_endpoint_summary, SEEDS
from experiments.formal.s3_initial_distribution.plot_results import pair_vectors


def test_endpoint_does_not_average_only_surviving_gaussian_predictions():
    """A failed Gaussian trajectory makes the seed/population endpoint unavailable, with 179 survivors diagnostic only."""
    metrics = {f'GIFT_seed_{seed}': {'full_relative_l2': np.ones((180, 7))} for seed in SEEDS}
    metrics[f'GIFT_seed_{SEEDS[0]}']['full_relative_l2'][0, -1] = np.nan
    result = gift_endpoint_summary(metrics, 'GIFT', retain_failures=True)
    assert result['mean'] is None
    assert result['values'][str(SEEDS[0])] is None
    assert result['finite_subset_diagnostics'][str(SEEDS[0])]['finite_count'] == 179
    with pytest.raises(ValueError):
        gift_endpoint_summary(metrics, 'GIFT')


def test_complete_endpoint_matches_existing_mean():
    """A complete three-seed endpoint uses the existing seed mean and sample SD conventions."""
    metrics = {f'GIFT_seed_{seed}': {'full_relative_l2': np.ones((180, 7))*i}
               for i, seed in enumerate(SEEDS, 1)}
    result = gift_endpoint_summary(metrics, 'GIFT', retain_failures=True)
    assert result['mean'] == 2 and result['sample_sd'] == 1


def test_vector_composition_preserves_sources_and_separates_ids(tmp_path):
    """Pairing SVG panels must preserve source bytes and prefix internal IDs without overwriting an output."""
    content = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80">
    <defs><clipPath id="clip"><path d="M0 0L10 10"/></clipPath></defs>
    <g clip-path="url(#clip)"><text x="3" y="4">sample</text></g></svg>'''
    left, right, target = (tmp_path/name for name in ('a.svg', 'b.svg', 'paired.svg'))
    left.write_bytes(content)
    right.write_bytes(content)
    result = pair_vectors(left, right, target)
    assert left.read_bytes() == right.read_bytes() == content
    root = ET.parse(target).getroot()
    ids = [item.get('id') for item in root.iter() if item.get('id')]
    assert ids == ['panel0_clip', 'panel1_clip']
    # Both panels are placed by vector translation with unique ID prefixes, so the
    # sources are never redrawn or rewritten by composition.
    assert len(ids) == len(set(ids))
    assert 'translation' in result['operation'] and 'vector' in result['operation']
    with pytest.raises(FileExistsError):
        pair_vectors(left, right, target)


@pytest.mark.parametrize("historical_hash", [False, True])
def test_s3_route_writes_separate_population_and_all_methods(tmp_path, monkeypatch, historical_hash):
    """Explicit synthetic routing fixture; no real model or scientific data used."""
    import json
    from types import SimpleNamespace
    import h5py
    from experiments.formal.w1_recursive_prediction import run
    from gift import paths as paths_module, gaussian_package
    from training import weight_files
    from training.checkpoints import digest_file
    from gift.prediction_cohorts import GAUSSIAN, partitions

    data = tmp_path/'data'
    observed = data/'s3_gaussian/trajectories.h5'
    observed.parent.mkdir(parents=True)
    observed.write_bytes(b'synthetic routing fixture, never a training dataset')
    weights = tmp_path/'weights.pt'
    weights.write_bytes(b'synthetic routing fixture, never a model')
    output = tmp_path/'result'
    monkeypatch.setattr(paths_module, 'data_root', lambda _: data)
    monkeypatch.setattr(paths_module, 'checkpoint_root', lambda _: tmp_path/'artifacts')
    monkeypatch.setattr(gaussian_package, 'validate_input', lambda *_, **__: {})
    args = SimpleNamespace(project_root=tmp_path, low_model=weights, fno2d_model=weights,
        fno3d_model=weights, uno_model=weights, unet_model=weights, output=output,
        device='cpu', gift_execution='eager', gift_batch_size=16, fno2d_batch_size=10,
        fno3d_batch_size=5, uno_batch_size=4, unet_batch_size=4, skip_plots=True)
    monkeypatch.setattr(run, 'parse_args', lambda: args)
    monkeypatch.setattr(run, 'resolve_regimes', lambda _, paths: {
        family: (paths, {seed: weights for seed in SEEDS}) for family in ('GIFT', 'GIFT-Lite')})
    monkeypatch.setattr(weight_files, 'load_weights', lambda _: dict(fresh_training=True,
        formal_configuration={'data_profile': 'gaussian'}, run_provenance={
            'identity': {'data': {'sha256': digest_file(observed) if historical_hash else None,
                                  'cohort': GAUSSIAN, 'training_ids': partitions(GAUSSIAN)['training']}}}))
    if not historical_hash:
        monkeypatch.setattr('training.checkpoints.digest_file',
                            lambda _: pytest.fail('ordinary S3 weights should not scan the dense file'))
    def session(*_, **__):
        """Provide an in-memory numerical-call shim so this test checks routing without running trained models."""
        output.mkdir()
        return SimpleNamespace(output=output, call=lambda key, fn, *a, **kw: fn(*a, **kw), finish=lambda: None,
            save_prediction=lambda group, key, prediction: group.create_dataset("prediction", data=prediction))
    monkeypatch.setattr(run, 'start_experiment', session)
    ids, times = np.arange(2260,2440), run.REPORT_TIMES
    truth = np.ones((180,7,64,64), dtype=np.float32)
    context = np.ones((180,1,64,64), dtype=np.float32)
    finite = np.ones((180,150), dtype=bool)
    monkeypatch.setattr(run, 'load_n64_long', lambda _: (ids,times,truth))
    monkeypatch.setattr(run, 'load_fno_test_dt0p02', lambda *_: (ids,None,context,times,truth))
    monkeypatch.setattr(run, 'load_gift_models', lambda *_: (None,None,None))
    monkeypatch.setattr(run, 'rollout_gift', lambda **_: SimpleNamespace(prediction=truth,
        failure_step=np.full(180,-1), finite_by_time=np.ones((180,7),dtype=bool),
        failure_reason=['']*180, correction={}, runtime={}))
    audit = {'finite_trajectory_count_by_future_step':[180]*150,
             'finite_by_trajectory_future_step':finite.tolist()}
    monkeypatch.setattr(run, 'load_fno_models', lambda *_: (None,None,None,None))
    monkeypatch.setattr(run, 'predict_fno2d', lambda *_, **__: (truth,audit))
    monkeypatch.setattr(run, 'predict_fno3d', lambda *_, **__: (truth,audit,None))
    monkeypatch.setattr(run, 'load_uno', lambda *_: (None,None))
    monkeypatch.setattr(run, 'load_unet', lambda *_: (None,None))
    monkeypatch.setattr(run, 'baseline_prediction', lambda **_: (truth,finite,np.full(180,-1)))
    monkeypatch.setattr(run, 'metric_arrays', lambda *_: {'full_relative_l2':np.zeros((180,7))})
    run.main(experiment='S3')
    report = json.loads((output/'report.json').read_text())
    assert report['experiment']=='S3' and report['figure']['keyframes']['trajectory_id']==2265
    assert report['populations']=={'test_2260_2439':{'trajectory_ids':[2260,2439],'count':180}}
    with h5py.File(output/'raw/predictions.h5') as handle:
        assert handle.attrs['schema']=='gift.formal.S3.raw.v2'
        assert handle['trajectory_ids'][:].tolist()==ids.tolist()
        assert all(group in handle for _,_,group in run.METHOD_CONFIGS)


def test_single_regime_keyframes_preserve_failed_sample_and_global_order(tmp_path):
    """A single-regime plot must keep NaN failures, visible failure labels and the shared method ordering."""
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from experiments.formal.w1_recursive_prediction import plot_keyframes as renderer
    methods = tuple(spec for spec in renderer.METHODS if spec["label"] == "GIFT")
    truth = np.ones((4, 64, 64), dtype=np.float32)
    prediction = truth.copy()
    prediction[1:] = np.nan
    path = tmp_path / "predictions.h5"
    with h5py.File(path, "x") as handle:
        handle.attrs["schema"] = "gift.formal.S3.raw.v2"
        handle.create_dataset("trajectory_ids", data=[2265])
        handle.create_dataset("absolute_times", data=renderer.KEY_TIMES)
        handle.create_dataset("truth", data=truth[None])
        group = handle.create_group(methods[0]["group"])
        group.create_dataset("prediction", data=prediction[None])
        group.create_dataset("full_relative_l2", data=[[0, np.nan, np.nan, np.nan]])
    before = renderer.METHODS
    observed, predictions, residuals, errors, positions = renderer.load_keyframes(
        path, 2265, experiment="S3", methods=methods)
    assert renderer.METHODS == before
    assert np.isnan(predictions["gift"][1:]).all()
    assert np.isnan(residuals["gift"][1:]).all()
    renderer.save_source_data(tmp_path, trajectory_id=2265, time_indices=positions,
        truth=observed, predictions=predictions, residuals=residuals, errors=errors, methods=methods)
    with np.load(tmp_path / "source_fields.npz", allow_pickle=False) as saved:
        assert saved["field_finite_gift"].tolist() == [True, False, False, False]
        assert np.isnan(saved["prediction_gift"][1:]).all()
    figure, axis = plt.subplots()
    renderer.draw_mesh(axis, prediction[1], cmap=renderer.FIELD_CMAP,
                       norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10))
    assert any(text.get_text() == "Prediction failed" for text in axis.texts)
    plt.close(figure)
