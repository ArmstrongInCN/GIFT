"""Append a verified S4 population, preserving every existing numeric file.

Metadata backups are mandatory and stay outside the Zenodo package. No existing
data directory is replaced, moved or deleted. Metadata are committed last.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT/'src'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
from gift.gaussian_package import validate_input
from gift.prediction_cohorts import GAUSSIAN, partitions
from scripts.verify_data import verify, sha256


DESCRIPTION = '''

## S4: Gaussian random-field initial conditions / 高斯随机场初值

`s4_gaussian/trajectories.h5` contains 1,220 independent N64 trajectories,
sampled at t=0, 0.02, ..., 10 (501 frames). IDs are appended to the original
population: training 1220–2219, validation 2220–2259, test 2260–2439.
The `validation` group contains checkpoint-validation IDs 2220–2239;
`validation_aux` contains 2240–2259. Neither validation nor test trajectories
enter gradient updates. `lite.h5` is an exact subset: training 1220–1269 and
validation 2220–2239, with no test fields.

初值为周期、空间相关的平滑零均值高斯随机场，而非独立像素白噪声。
For a real independent standard-normal grid z, the Fourier filter is
h(k)=exp(−0.4²|k|²/4), with h(0)=0. The field is
IFFT[a h(k) FFT(z)], where a=σ N/sqrt(sum h²), N=64 and
σ=1.0908839162005806. This fixed ensemble amplitude is determined only by
the original 1,000 training initial fields. Individual draws are not rescaled,
clipped, rejected or selected by test outcomes. PCG64 uses
SeedSequence([2026091601, trajectory_id]); draw order and batching do not change
the initial field. The PDE, float32/complex64 ETDRK4 solver, internal dt=0.005
and stored dt=0.02 match the four-vortex population.

Each HDF5 group contains `trajectory_index` (int64), `time` (float64) and
`vorticity` (float32, [trajectory,time,x,y]). No analytic derivative labels or
four-vortex parameter arrays are supplied for Gaussian data. `run.json` and
`COMPLETE.json` bind the simulator, initial-condition law and completed output;
the nested `manifest.json` records byte hashes, decoded-field hashes and splits.
The root manifest also covers all five files. License: CC BY 4.0, unchanged.
'''


def append(package, dataset, backup):
    package, dataset = Path(package).resolve(strict=True), Path(dataset).resolve(strict=True)
    backup = Path(backup).resolve()
    destination = dataset/'s4_gaussian'
    if (destination.exists()
            or backup.exists() or dataset == backup or dataset in backup.parents):
        raise ValueError('Use an unextended data package and a new external metadata-backup directory')
    for name in ('trajectories.h5', 'lite.h5'):
        validate_input(package.parent, package/name, full=True)
    verify(dataset)
    previous = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    nested = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    metadata = ('README.md', 'DATA_DICTIONARY.md', 'splits.json', 'schema.json', 'manifest.json')
    backup.mkdir(parents=True, exist_ok=False)
    for name in metadata:
        shutil.copy2(dataset/name, backup/name)
    # copytree refuses an existing destination; its contents are all new files.
    shutil.copytree(package, destination)
    for name in ('README.md', 'DATA_DICTIONARY.md'):
        with (dataset/name).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(DESCRIPTION)
    splits = json.loads((backup/'splits.json').read_text(encoding='utf-8'))
    splits.setdefault('additional_populations', {})[GAUSSIAN] = partitions(GAUSSIAN)
    schema = json.loads((backup/'schema.json').read_text(encoding='utf-8'))
    schema['files'].update({'s4_gaussian/'+name: value for name, value in nested['observation_schema'].items()})
    for name, value in (('splits.json', splits), ('schema.json', schema)):
        (dataset/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    records = []
    for record in previous['files']:
        record = dict(record)
        file = dataset/record['path']
        if record['path'] in metadata:
            record.update(bytes=file.stat().st_size, sha256=sha256(file))
        elif file.stat().st_size != record['bytes'] or sha256(file) != record['sha256'].lower():
            raise ValueError('An original data file changed during append')
        records.append(record)
    for file in sorted(destination.iterdir()):
        records.append(dict(path=file.relative_to(dataset).as_posix(), bytes=file.stat().st_size,
            sha256=sha256(file), data_license='CC-BY-4.0',
            category='scientific_input' if file.suffix=='.h5' else 'metadata'))
    updated = dict(previous, files=records, file_count=len(records), total_bytes=sum(r['bytes'] for r in records),
                   gaussian_population='s4_gaussian', gaussian_observations_lossless=True)
    (dataset/'manifest.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result = verify(dataset, full_array_scan=True)
    result['original_numeric_files_unchanged'] = True
    (backup/'APPEND_COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--metadata-backup', type=Path, required=True)
    args = parser.parse_args()
    append(args.package, args.dataset, args.metadata_backup)


if __name__ == '__main__':
    main()
