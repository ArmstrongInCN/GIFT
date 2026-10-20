"""Generate the S3 population once; preserve own-attempt ETDRK4 checkpoints.

Only the initial-condition distribution differs from the W1 physical solver.
The disjoint training, validation and test populations are fixed before
generation. Select a new output directory or resume its saved solver state
with the same generation parameters; source/environment metadata is descriptive.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gift.gaussian_initial import initial_hat, specification
from gift.prediction_cohorts import GAUSSIAN, partitions
from scripts import generate_data as generation


def make_plan(args):
    # S3 partitions the Gaussian cohort into the formal disjoint splits.
    """Build the fixed S3 split and 0.02-second observation schedule on the N64 grid.

    The checkpoint-validation rows and auxiliary validation rows stay separate.
    Subsets and short pilots are labelled and do not become a full S3 package.
    """
    split = partitions(GAUSSIAN)
    # validation contains checkpoint-selection IDs 2220-2239; validation_aux
    # contains IDs 2240-2259. Both remain separate from training and test rows.
    ids_by_group = {"training": split['training'],
                   "validation": split['checkpoint_validation'],
                   "validation_aux": split['validation'][20:], "test": split['test']}
    chosen = generation.parse_ids(args.subset)
    groups, found = [], set()
    for name, ids in ids_by_group.items():
        ids = [x for x in ids if chosen is None or x in chosen]
        if not ids:
            continue
        found.update(ids)
        final_step = 2000 if args.pilot_steps is None else args.pilot_steps
        steps = sorted(set([0, final_step, *range(0, final_step + 1, 4)]))
        groups.append(dict(name=name, split=name, grid=64, batch_size=args.batch_size,
                           steps=steps, ids=ids))
    if not groups or (chosen is not None and found != chosen):
        raise ValueError("Subset contains IDs outside the prespecified S3 population")
    # prediction_cohort and initial_condition record the Gaussian law and cohort,
    # so the later package and append steps can bind the exact initial-condition
    # distribution without re-deriving it from the raw fields.
    return dict(schema='gift.gaussian-generation.v1', dataset='s3-gaussian',
        prediction_cohort=GAUSSIAN, initial_condition=specification(),
        pilot=args.pilot_steps is not None or args.subset is not None,
        subset=args.subset, dt=generation.DT, device=args.device, groups=groups,
        partitions=split,
        physical_protocol=dict(equation='omega_t+u.grad(omega)=0.01*laplacian(omega)-4*cos(4*y)',
            domain='[0,2*pi)^2', internal_dt=0.005, output_dt=0.02,
            real_dtype='float32', complex_dtype='complex64', grid=64, padding_grid=99,
            solver='full-spectrum Fourier pseudospectral ETDRK4', state_mask_applied=False),
        sources={name: generation.sha256(ROOT/name) for name in (
            'scripts/generate_gaussian_data.py', 'scripts/generate_data.py',
            'src/gift/gaussian_initial.py', 'src/gift/prediction_cohorts.py',
            'src/gift/data_splits.py', 'src/even_full_spectrum_ns.py')})


# Build the spectral initial state for a batch directly from the Gaussian
# initial-condition law; no reference solution or trained weights are consulted.
def build_state(group, start, end, device):
    """Create complex Fourier initial states for IDs in the current Gaussian batch.

    The seeded initial-condition law determines each field independently of
    batch order; no reference trajectory or trained model is read.
    """
    import torch
    return torch.from_numpy(initial_hat(group['ids'][start:end])).to(device)


def main(argv=None):
    """Plan or resume Gaussian initial-condition integration through the common NS solver."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda','cuda:0'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--subset', help='Diagnostic selection, never a full S3 data package')
    parser.add_argument('--pilot-steps', type=int)
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error('Batch/checkpoint intervals must be positive')
    if args.pilot_steps is not None and not 0 <= args.pilot_steps <= 2000:
        parser.error('Pilot steps must be between 0 and 2000')
    if args.stop_after_steps is not None and args.stop_after_steps < 0:
        parser.error('Stop boundary must be nonnegative')
    plan = make_plan(args)
    generation.safe_output(args.output)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    return generation.execute(args, plan, initial_state_builder=build_state)


if __name__ == '__main__':
    raise SystemExit(main())
