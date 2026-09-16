"""Explicit S4 input gates, separate from the unchanged four-vortex package."""
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np

from .canonical_package import _digest, _member
from .gaussian_initial import specification
from .prediction_cohorts import GAUSSIAN, partitions

DIRECTORY = 's4_gaussian'
DENSE_FILE = DIRECTORY + '/trajectories.h5'
LITE_FILE = DIRECTORY + '/lite.h5'


def group_ids(*, lite=False):
    split = partitions(GAUSSIAN)
    groups = dict(training=split['gift_lite_training'] if lite else split['training'],
                  validation=split['checkpoint_validation'])
    if not lite:
        groups.update(validation_aux=split['validation'][20:],test=split['test'])
    return groups


def validate_input(base, path, *, full=False):
    """Check the exact file identity; scan only training/validation observations.

    Test fields are never read by this training gate. Whole-package verification
    independently checks their finite values before any model is run.
    """
    base, path = Path(base).resolve(strict=True), Path(path).resolve(strict=True)
    relative = path.relative_to(base).as_posix()
    if relative not in (DENSE_FILE,LITE_FILE) or path != _member(base,relative):
        raise ValueError('Gaussian training requires its explicit S4 package member')
    root = base/DIRECTORY
    manifest_path = _member(base,DIRECTORY+'/manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema')!='gift.gaussian-data-package.v1'
            or manifest.get('status')!='complete'
            or manifest.get('initial_condition')!=specification()
            or manifest.get('partitions')!=partitions(GAUSSIAN)):
        raise ValueError('Gaussian population specification/partitions differ')
    records={row['path']:row for row in manifest['files']}
    if len(records)!=len(manifest['files']) or set(records)!={'trajectories.h5','lite.h5','run.json','COMPLETE.json'}:
        raise ValueError('Gaussian package file inventory differs')
    for name,record in records.items():
        item = _member(root,name)
        digest = record['sha256']
        if (type(record['bytes']) is not int or record['bytes'] < 1
                or not isinstance(digest,str) or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)
                or item.stat().st_size!=record['bytes']):
            raise ValueError('Gaussian package file size differs')
        if (name in ('run.json','COMPLETE.json') or (full and item==path)) and _digest(item)!=record['sha256']:
            raise ValueError('Gaussian package consumed-file hash differs')
    run=json.loads((root/'run.json').read_text(encoding='utf-8'))
    receipt=json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
    plan=run['binding']['plan']
    actual_binding=hashlib.sha256(json.dumps(run['binding'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (receipt['status']!='COMPLETE_GENERATION' or plan['pilot'] or plan['subset'] is not None
            or plan['initial_condition']!=specification() or plan['partitions']!=partitions(GAUSSIAN)
            or receipt['binding_sha256']!=run['binding_sha256']
            or actual_binding!=run['binding_sha256']
            or receipt['attempt_id']!=run['attempt_id']
            or receipt['data_sha256']!=records['trajectories.h5']['sha256']):
        raise ValueError('Gaussian generation is incomplete, diagnostic or unbound')
    expected=group_ids(lite=relative==LITE_FILE)
    with h5py.File(path,'r') as handle:
        if (handle.attrs.get('prediction_cohort')!=GAUSSIAN
                or handle.attrs.get('trajectory_id_scheme')!='canonical'
                or bool(handle.attrs.get('pilot',True))
                or handle.attrs.get('status')!='COMPLETE_GENERATION'):
            raise ValueError('Gaussian observations are not a completed formal population')
        if relative==LITE_FILE and handle.attrs.get('parent_sha256')!=records['trajectories.h5']['sha256']:
            raise ValueError('Reduced-data observations have a different parent')
        if set(handle)!=set(expected):
            raise ValueError('Gaussian observation groups differ')
        for name,ids in expected.items():
            field=handle[name+'/vorticity']
            if (handle[name+'/trajectory_index'][:].tolist()!=ids
                    or field.shape!=(len(ids),501,64,64) or field.dtype!=np.dtype('float32')
                    or not np.allclose(handle[name+'/time'][:],np.arange(501)*.02,rtol=0,atol=2e-12)):
                raise ValueError('Gaussian identities, observation shape or times differ')
            if full and name in ('training','validation'):
                for row in range(len(ids)):
                    if not np.isfinite(field[row]).all():
                        raise ValueError('Nonfinite Gaussian training/validation data')
    return dict(profile='gaussian',cohort=GAUSSIAN,file=relative,
        sha256=records[path.name]['sha256'] if full else None,bytes=path.stat().st_size,
        manifest_sha256=_digest(manifest_path),trajectory_ids=expected['training'],
        validation_ids=expected['validation'],full_input_checks_performed=full,
        published_dataset_identity=False,numerical_reproduction_verified=False)
