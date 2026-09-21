"""Package a completed S4 simulation without interpolation or new integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
for item in (ROOT,ROOT/'src'):
    if str(item) not in sys.path:
        sys.path.insert(0,str(item))
import h5py
import numpy as np
from gift.gaussian_initial import specification
from gift.gaussian_package import group_ids, validate_input
from gift.prediction_cohorts import GAUSSIAN, partitions
from scripts.generate_data import sha256, canonical, write_json_new, safe_output


def prepare(generation,output):
    # Package one completed S4 Gaussian generation into the released s4_gaussian
    # directory; no interpolation or new integration, and every source hash must
    # still match the generation plan's recorded generator/solver sources.
    generation=Path(generation).resolve(strict=True)
    output=safe_output(output)
    if output.name!='s4_gaussian' or output.exists():
        raise ValueError('Use a new external output directory named s4_gaussian')
    run=json.loads((generation/'run.json').read_text(encoding='utf-8'))
    receipt=json.loads((generation/'COMPLETE.json').read_text(encoding='utf-8'))
    plan=run['binding']['plan']
    binding=hashlib.sha256(canonical(run['binding']).encode()).hexdigest()
    if (receipt['status']!='COMPLETE_GENERATION' or plan['pilot'] or plan['subset'] is not None
            or plan['schema']!='gift.gaussian-generation.v1'
            or plan['initial_condition']!=specification() or plan['partitions']!=partitions(GAUSSIAN)
            or binding!=run['binding_sha256'] or binding!=receipt['binding_sha256']
            or run['attempt_id']!=receipt['attempt_id']
            or sha256(generation/'data.h5')!=receipt['data_sha256']):
        raise ValueError('Require one completed, full-population Gaussian generation')
    for relative,expected in plan['sources'].items():
        if sha256(ROOT/relative)!=expected:
            raise ValueError('Gaussian generation source changed before packaging')
    schemas={}
    with h5py.File(generation/'data.h5','r') as handle:
        if handle.attrs['status']!='COMPLETE_GENERATION' or handle.attrs['prediction_cohort']!=GAUSSIAN:
            raise ValueError('Incomplete Gaussian HDF5')
        for name,ids in group_ids().items():
            field=handle[name+'/vorticity']
            if (handle[name+'/trajectory_index'][:].tolist()!=ids
                    or field.shape!=(len(ids),501,64,64)
                    or not np.array_equal(handle[name+'/time'][:],np.arange(501)*.02)):
                raise ValueError('Gaussian population/time dimensions differ')
            digest=hashlib.sha256()
            for row in range(len(ids)):
                value=np.asarray(field[row])
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite generated trajectory; no sample is omitted')
                digest.update(value.tobytes())
            schemas['/'+name]={'trajectory_ids':ids}
            schemas['/'+name+'/vorticity']=dict(shape=list(field.shape),dtype=str(field.dtype),decoded_sha256=digest.hexdigest())
    output.mkdir(parents=True,exist_ok=False)
    for source,target in (('data.h5','trajectories.h5'),('run.json','run.json'),('COMPLETE.json','COMPLETE.json')):
        with (generation/source).open('rb') as incoming,(output/target).open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,8*1024*1024)
        if sha256(generation/source)!=sha256(output/target):
            raise ValueError('Package copy is not byte-exact')
    lite_schema={}
    with h5py.File(output/'trajectories.h5','r') as original,h5py.File(output/'lite.h5','x') as lite:
        for key,value in original.attrs.items():
            lite.attrs[key]=value
        lite.attrs['parent_sha256']=receipt['data_sha256']
        # GIFT-Lite is an exact row sub-selection: the first 50 training IDs and
        # the 20 checkpoint-validation IDs, with no test fields copied.
        lite.attrs['derivation']='exact first 50 training and 20 checkpoint-validation rows'
        for name,ids in group_ids(lite=True).items():
            group=lite.create_group(name)
            group.create_dataset('trajectory_index',data=np.asarray(ids,dtype=np.int64))
            group.create_dataset('time',data=original[name+'/time'][:])
            field=group.create_dataset('vorticity',shape=(len(ids),501,64,64),dtype='f4',
                chunks=(1,1,64,64),compression='lzf',shuffle=True)
            digest=hashlib.sha256()
            for row in range(len(ids)):
                values=original[name+'/vorticity'][row]
                field[row]=values
                if not np.array_equal(field[row],values):
                    raise ValueError('Reduced-data copy changes observations')
                digest.update(values.tobytes())
            lite_schema['/'+name]={'trajectory_ids':ids}
            lite_schema['/'+name+'/vorticity']=dict(shape=list(field.shape),dtype=str(field.dtype),decoded_sha256=digest.hexdigest())
    # The package manifest binds every file by byte hash and records the decoded
    # vorticity hash per group, so the released artifact is fully self-verifying.
    manifest=dict(schema='gift.gaussian-data-package.v1',status='complete',
        initial_condition=specification(),partitions=partitions(GAUSSIAN),
        physical_protocol=plan['physical_protocol'],
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(output.iterdir())],
        observation_schema={'trajectories.h5':schemas,'lite.h5':lite_schema},
        observations_modified=False,all_generated_observations_finite=True)
    write_json_new(output/'manifest.json',manifest)
    for name in ('trajectories.h5','lite.h5'):
        validate_input(output.parent,output/name,full=True)
    print(json.dumps(dict(status='PASS_COMPLETE_GAUSSIAN_PACKAGE',trajectories=1220,
                         first_id=1220,last_id=2439,files=5)))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.generation,args.output)


if __name__=='__main__':
    main()
