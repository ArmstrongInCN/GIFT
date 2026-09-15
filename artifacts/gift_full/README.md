# Full-data GIFT trained weights

The generator and all three branches use the common 1,000-trajectory prediction
training set. The generator has 500 trajectory epochs, 31,500 gradient updates
and 501 affine fits. Each branch has 500 trajectory epochs and 45,200 updates.
Report these budgets as **500 + 500**, with generator training shared once.

`generator.pt` is the prerequisite for `gift_seed_20260820.pt`,
`gift_seed_20260821.pt` and `gift_seed_20260822.pt`. Each branch binds its exact
generator hash. The weights are terminal states; validation is monitoring only.
No test trajectory is used to choose a seed, epoch or model.

The completion receipts and `histories/` record the committed training work.
`TRAINING_COSTS.json` gives actual cumulative epoch times and their scope; these
are not controlled-load speed benchmarks. File hashes are listed in
`../CHECKPOINTS.json`.

Use these exports for quick evaluation. To train independently or continue your
own interrupted run, follow [the training protocol](../../docs/TRAINING_PROTOCOL.md)
and [checkpoint instructions](../../docs/CHECKPOINTS.md). Evaluation exports do
not replace a training run's optimizer, scheduler and random-state journal.
