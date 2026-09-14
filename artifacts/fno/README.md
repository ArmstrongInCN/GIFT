# FNO trained weights

These terminal checkpoints contain the trained model state and, for FNO-3D,
training-only normalization statistics. Both runs completed 500 epochs on the
same 1,000 trajectories. The measured update counts and cumulative epoch times
are in `training_cost.json`; checkpoint files retain their training identities.

Load the model implementation from the fixed upstream repository as described
in [the setup guide](../../docs/SETUP.md). These weights are for quick evaluation,
not a replacement for a run's optimizer/RNG checkpoint when continuing training.
Independent training commands and resume boundaries are described in
[the training protocol](../../docs/TRAINING_PROTOCOL.md).
