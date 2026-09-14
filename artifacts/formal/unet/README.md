# U-Net terminal weights

The checkpoint contains the native upstream U-Net after 500 epochs and 25,000
Adam updates on the declared 1,000 trajectories. Its 500-row training history,
terminal model/optimizer/scheduler state and full tensor equality were checked.
The external-loader path also passes a fresh four-step first-batch comparison
of initialization, samples, losses, all gradients and optimizer updates. That
short comparison is not an additional full training campaign.

Portable metadata does not change model tensors. `TRAINING_COST.json` reports
the sum of actual committed-epoch times, excluding checkpoint writing and setup.
Use the checkpoint for evaluation; use a new output directory to train from
scratch, or the same run directory and `--resume` to continue an interrupted run.

Obtain the upstream implementation separately as explained in
`docs/EXTERNAL_ADAPTATIONS.md`. These weights do not include its source code or
grant rights to that source. Model hashes are listed in `artifacts/CHECKPOINTS.json`.
