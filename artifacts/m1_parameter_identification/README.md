# Equation-identification generators

`gift_noise_000.pt`, `gift_noise_001.pt` and `gift_noise_010.pt` are the three
generators used by the equation-parameter identification experiment (M1), one per
observation-noise condition. Each was trained on the single local training
trajectory 0 and then frozen; the equation parameters are read out from that same
trajectory, exactly as the PDE-FIND and PINN-SR configurations consume it.

These weights are used only by that experiment. They are independent of the
GIFT-Lite clean generator in `../fixed_k21_n64_unmasked/` and of the full-data
generator in `../gift_full/`; a clean M1 generator is not interchangeable with a
prediction generator, and none of the three is a prediction model.

`manifest.json` records each file's condition, path, byte size and SHA-256, and
`../CHECKPOINTS.json` carries the same files in the repository-wide checkpoint
catalogue. The protocol and the per-configuration results are in `EXPERIMENTS.md`
section 4.3; the readout commands are in `docs/SETUP.md`.
