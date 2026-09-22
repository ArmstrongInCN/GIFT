# GIFT-Lite clean generator

`gift_main.pt` is the preserved clean generator for the reduced-data GIFT-Lite
regime. It was trained on the first 50 prediction trajectories (IDs 0-49) under
the multi-stage schedule in `docs/TRAINING_PROTOCOL.md`; reduced training data,
not a smaller network. The three reduced-data branches in `../gift/` share it as
a frozen prerequisite.

It is **not** the generator used by the equation-identification experiment: those
are three separate checkpoints under `../m1_parameter_identification/`, one per
noise condition. Neither set substitutes for the other, and neither substitutes
for the full-data generator in `../gift_full/generator.pt`.

The directory name is the one under which these weights were packaged.
`artifacts/CHECKPOINTS.json` and every `results/formal/` manifest bind this exact
path, so the directory has to keep its name.

See `../CHECKPOINTS.json` for the file hash, `../gift/TRAINING_COSTS.json` for
update counts and measured times, and `docs/CHECKPOINTS.md` for how the two GIFT
regimes differ. These are terminal evaluation weights: do not initialise a fresh
experiment or an unrelated run with them.
