"""Sparse, deliberately nonfinite fixtures test gates, not a scientific run."""
import hashlib
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gift.gaussian_package import group_ids, validate_input
from gift.gaussian_initial import specification
from gift.prediction_cohorts import GAUSSIAN, partitions
from training.checkpoints import digest_file


@pytest.fixture
def package(tmp_path):
    root=tmp_path/'s4_gaussian'
    root.mkdir()
    for name,lite in (('trajectories.h5',False),('lite.h5',True)):
        with h5py.File(root/name,'x') as handle:
            handle.attrs.update(prediction_cohort=GAUSSIAN,trajectory_id_scheme='canonical',
                                pilot=False,status='COMPLETE_GENERATION')
            if lite:
                handle.attrs['parent_sha256']=digest_file(root/'trajectories.h5')
            for group,ids in group_ids(lite=lite).items():
                handle.create_dataset(group+'/trajectory_index',data=np.asarray(ids,dtype='i8'))
                handle.create_dataset(group+'/time',data=np.arange(501)*.02)
                handle.create_dataset(group+'/vorticity',shape=(len(ids),501,64,64),dtype='f4',
                                      chunks=(1,1,64,64),fillvalue=np.nan)
    plan=dict(initial_condition=specification(),partitions=partitions(GAUSSIAN),pilot=False,subset=None)
    binding=dict(plan=plan,runtime={})
    sha=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'run.json').write_text(json.dumps(dict(binding=binding,binding_sha256=sha,attempt_id='fixture')),encoding='utf-8')
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_GENERATION',binding_sha256=sha,
        attempt_id='fixture',data_sha256=digest_file(root/'trajectories.h5'))),encoding='utf-8')
    manifest=dict(schema='gift.gaussian-data-package.v1',status='complete',initial_condition=specification(),
        partitions=partitions(GAUSSIAN),files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest_file(p)) for p in root.iterdir()])
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return tmp_path,root,manifest


def test_metadata_never_claims_completed_numerical_validation(package):
    base,root,_=package
    result=validate_input(base,root/'trajectories.h5',full=False)
    assert result['trajectory_ids']==list(range(1220,2220))
    assert result['sha256'] is None and not result['full_input_checks_performed']
    assert not result['numerical_reproduction_verified']
    lite=validate_input(base,root/'lite.h5',full=False)
    assert lite['trajectory_ids']==list(range(1220,1270))


def test_unwritten_truth_cannot_supply_training(package):
    base,root,_=package
    with pytest.raises(ValueError,match='Nonfinite'):
        validate_input(base,root/'lite.h5',full=True)


@pytest.mark.parametrize('change',['overlap','covariance','missing','duplicate','hash'])
def test_corrupt_gaussian_manifest_rejected(package,change):
    base,root,value=package
    if change=='overlap': value['partitions']['test'][0]=1220
    elif change=='covariance': value['initial_condition']['correlation_length']=.8
    elif change=='missing': value['files'].pop()
    elif change=='duplicate': value['files'].append(value['files'][0])
    else: value['files'][0]['sha256']='bad'
    (root/'manifest.json').write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError): validate_input(base,root/'lite.h5',full=False)


def test_baseline_budget_unchanged_for_gaussian():
    from training.baseline_control import _arguments,configuration
    for family in ('fno2d','fno3d','uno','unet'):
        args=_arguments(family,['--data-profile','gaussian'])
        assert args.data_file=='s4_gaussian/trajectories.h5'
        new=configuration(family,args)
        old=configuration(family,_arguments(family,[]))
        assert {k:v for k,v in new.items() if k!='data_profile'}=={k:v for k,v in old.items() if k!='data_profile'}


def test_gaussian_prediction_reads_appended_ids(tmp_path):
    from experiments.formal._shared.prediction_data import _observations
    path=tmp_path/'input.h5'
    with h5py.File(path,'x') as handle:
        handle.attrs.update(prediction_cohort=GAUSSIAN,trajectory_id_scheme='canonical')
        handle.create_dataset('test/trajectory_index',data=np.arange(2260,2440,dtype='i8'))
        handle.create_dataset('test/time',data=np.asarray([0.,.02]))
        handle.create_dataset('test/vorticity',shape=(180,2,64,64),dtype='f4',fillvalue=1.)
    ids,times,fields=_observations(path,'N64/test',64,cohort=GAUSSIAN)
    assert ids.tolist()==list(range(2260,2440)) and fields.shape==(180,2,64,64)
    with pytest.raises(ValueError,match='population'):
        _observations(path,'test',64)
