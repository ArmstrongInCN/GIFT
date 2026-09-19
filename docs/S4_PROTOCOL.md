# S4 — Initial-condition distribution / 初值分布对照

S4 repeats the M2 prediction protocol with a separate population of smooth,
spatially correlated Gaussian random-field initial conditions. Each population
has its own training and test data; this is not a train-on-one/test-on-another
distribution-transfer experiment. The completed M2 models and measurements are
the four-vortex reference, not initialization weights for S4.

## Data and fixed split

| Use | Trajectory IDs | Count |
|---|---|---:|
| Training | 1220–2219 | 1,000 |
| Checkpoint validation | 2220–2239 | 20 |
| Auxiliary validation | 2240–2259 | 20 |
| Test | 2260–2439 | 180 |
| GIFT-Lite training subset | 1220–1269 | 50 |

All IDs follow the original population's last ID, 1219. Every trajectory is
generated under the same prescribed distribution before any model training.
Only the training population participates in gradient updates. Full-data GIFT
and the baselines use terminal weights; GIFT-Lite retains M2's validation-based
selection. Test outcomes never select a checkpoint or alter the distribution.

The periodic zero-mean Gaussian field is a linear Fourier transform of independent
standard-normal samples. With h(k)=exp(−0.4²|k|²/4), h(0)=0, its Fourier coefficients
are a h(k) FFT(z), where a=σN/sqrt(sum h²), N=64 and σ=1.0908839162005806.
This fixed ensemble amplitude uses only original training initial fields.
Individual realizations are not normalized, clipped or rejected. PCG64 is seeded
by SeedSequence([2026091601, trajectory_id]). The physical equation and numerical
solver match M2: periodic [0,2π)², viscosity0.01, forcing−4cos(4y), full-spectrum
ETDRK4, internal dt0.005, output dt0.02, N64, padding99, float32/complex64.

## Independent generation and training

Generate a new output directory, package the completed data, then set
`GIFT_DATA_ROOT` to its parent data-package directory. Data stay outside GitHub.

```shell
python -m scripts.generate_gaussian_data --output ../runs/s4_generation --device cuda --execute
python -m scripts.prepare_gaussian_package --generation ../runs/s4_generation --output ../gift-data/s4_gaussian
```

For interrupted generation, add `--resume` with the same output. Diagnostic
`--subset` or `--pilot-steps` outputs cannot qualify as the complete S4 population.
Generation's committed state includes solver state and already stored frames.

Run each model separately, after the external repositories have been installed
as documented in [EXTERNAL_ADAPTATIONS.md](EXTERNAL_ADAPTATIONS.md):

```shell
python -m scripts.run_training gift_generator --run-training --dataset ../gift-data/s4_gaussian/trajectories.h5 --output ../runs/s4_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../gift-data/s4_gaussian/trajectories.h5 --low-model ../runs/s4_generator/model.pt --seed 20260820 --output ../runs/s4_gift_20260820
python -m scripts.run_training gift_low --run-training --data-profile gaussian --output ../runs/s4_lite_generator
python -m scripts.run_training gift_branch --run-training --data-profile gaussian --low-model ../runs/s4_lite_generator/gift_main.pt --seed 20260820 --output ../runs/s4_lite_20260820
python -m scripts.run_training fno2d --run-training --data-profile gaussian --output ../runs/s4_fno2d
python -m scripts.run_training fno3d --run-training --data-profile gaussian --output ../runs/s4_fno3d
python -m scripts.run_training uno --run-training --data-profile gaussian --output ../runs/s4_uno
python -m scripts.run_training unet --run-training --data-profile gaussian --output ../runs/s4_unet
```

Repeat each GIFT branch command independently for seeds20260821 and20260822.
Full-data generator and branch each receive500epochs, reported separately.
All four baselines receive500epochs. Data size, model structure, loss, optimizer,
batch size, rollout lengths, random seeds and selection rules match M2; only
the observation population and its IDs change. GIFT-Lite keeps the original
50-trajectory budget. See [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md) for update
counts and timing scope. No external algorithm code is modified for S4.

For full-data GIFT and the Lite generator, replace `--run-training` with
`--resume` to continue the same attempt. For Lite branches and the four baselines,
retain `--run-training` and add `--resume`. Use the same output, source, data and
runtime; a published model is not an optimizer-state continuation checkpoint.

## Evaluation and figure protocol

S4 uses the same t=5 anchor, baseline46-frame context and150-step future through
t=8 as M2. All180 test trajectories are evaluated. Failed predictions, if any,
remain in numerical records with finite counts; they are never silently omitted
from a reported full-population comparison.

`python -m scripts.run_experiment S4 -- --output ../runs/S4` uses the published
checkpoint layout under `artifacts/s4_gaussian/`. Each model path can instead be
specified explicitly using the same overrides as M2. Numerical evaluation and
plotting are independent; use `--skip-plots` then:

```shell
python -m experiments.formal.s4_initial_distribution.plot_results --result-dir ../runs/S4 --output-dir ../runs/S4_figures
```

The paired curve preserves M2 on the left and places Gaussian measurements on
the right, using the same visual design. The original M2 keyframe plate is reused
unchanged; a separate Gaussian plate uses prespecified ID2265 and GIFT seed20260820,
matching the sixth test-row position of the original ID1045. All report figures
are SVG. Axis/color ranges expand only when necessary; no outcome-dependent
trajectory selection or method-specific color scaling is allowed.

## Completed run and its limits

The completed S4 run is published in compact form at
`results/formal/S4_initial_distribution/`: summary tables, the paired curve, the
Gaussian keyframe plate and their source arrays, with `published.json` binding the
measurement, executed sources and released inputs. The measured comparison and its
limitations are reported in [EXPERIMENTS.md](../EXPERIMENTS.md) section 5.4.

Every test trajectory is retained. If a method's later report times no longer
cover all 180 test trajectories because some predictions became non-finite, its
curve is stopped at the last complete time and annotated on the panel rather than
drawn from a finite subset, its per-time `finite_count` is carried in
`summary/metrics.csv`, and the endpoint summary records
`status: nonfinite_predictions` instead of a full-population mean. Do not read a
finite-subset mean as the whole-population result.

Keyframe colour limits are prespecified constants and are never fitted to the
plotted trajectory. The four-vortex plate keeps the paper's ±19 (scalar field) and
±21 (signed residual). The Gaussian population declares its own prespecified pair,
**±25 for both quantities**, because its reference field genuinely exceeds ±19: the
prespecified example trajectory 2265 reaches about 24.06. A prespecified range the
selected keyframe would exceed makes the renderer refuse the figure rather than
widen the scale, and the two limits are recorded in `figure_manifest.json`.
