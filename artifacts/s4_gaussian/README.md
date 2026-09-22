# Weights for the Gaussian initial-condition regime (S4)

S4 repeats the same prediction protocol on a separate population of smooth
Gaussian random-field initial conditions, so it uses its own weights:

- `gift_full/generator.pt` and `gift_full/gift_seed_SEED.pt` — full-data GIFT.
- `gift_lite/generator.pt` and `gift_lite/gift_seed_SEED.pt` — reduced-data GIFT-Lite.
- `fno2d/`, `fno3d/`, `unet/` — one ordinary `.pt` file per comparison model.
- `uno/` — `weights.json` plus numbered `.part` files; keep this folder together and do not load a single part on its own.

Every model in this directory was trained from scratch on the Gaussian
trajectories and is a terminal state, not an initialisation for further training.
The S4 evaluation rejects any model whose recorded input population, freshness or
frozen-generator binding differs, so a four-vortex model cannot silently stand in
for one of these.

`TRAINING_COSTS.json` reports each run's declared budget and measured committed
interval with an explicit timing scope; the scopes differ between model families
and are not pooled into one comparable measurement. Data generation, splits,
training and evaluation commands are in `docs/S4_PROTOCOL.md`, file hashes in
`../CHECKPOINTS.json`, and the measured numbers in `EXPERIMENTS.md` section 5.4.
