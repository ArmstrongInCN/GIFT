# GIFT-Lite trained weights

The three reduced-data branch checkpoints use seeds 20260820, 20260821 and 20260822 and share
the clean generator at `../fixed_k21_n64_unmasked/gift_main.pt`. Noisy generators
for M1 are under `../m1_parameter_identification/`. All six models were trained
from fresh initialization under the protocol in `docs/TRAINING_PROTOCOL.md`.

The complete training histories and saved boundaries were checked before
packaging. Export changes portable metadata, not numerical model tensors.
The branch prerequisite link identifies the packaged low generator; its training
artifact hash and tensor identity remain recorded separately. Terminal weights
are for evaluation, not for continuing a different training attempt.

`TRAINING_COSTS.json` records update counts and measured timings. Clean generator
pretraining is shared and counted once. Branch end-to-end timings include
data/RHS preparation, training, validation, selection and export, but exclude
shared pretraining. Separately recorded per-stage committed times include the
training passes and validation, not preparation, checkpoint IO or export.
Do not conflate these scopes or count the shared generator three times.

See `../CHECKPOINTS.json` for file hashes. Follow independent fresh-training or
same-run resume commands in `docs/SETUP.md`; do not initialize a fresh experiment
with these downloaded trained weights.
