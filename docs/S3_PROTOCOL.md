# S3 — Initial-condition distribution / 初值分布对照

S3 compares four-vortex and smooth, spatially correlated Gaussian random-field
initial conditions. Each distribution has its own models, 1,000 training
trajectories and 180 test trajectories. GIFT receives one initial state; the
baselines receive 46 historical frames and forecast 150 steps at dt = 0.02.
This within-distribution comparison is independent of the N300, single-state,
dt = 0.04 W1/W2 protocol.

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
and the baselines use terminal weights; GIFT-Lite uses validation-based selection.
Test outcomes never select a checkpoint or alter the distribution.

The periodic zero-mean Gaussian field is a linear Fourier transform of independent
standard-normal samples. With h(k)=exp(−0.4²|k|²/4), h(0)=0, its Fourier coefficients
are a h(k) FFT(z), where a=σN/sqrt(sum h²), N=64 and σ=1.0908839162005806.
This fixed ensemble amplitude uses only original training initial fields.
Individual realizations are not normalized, clipped or rejected. PCG64 is seeded
by SeedSequence([2026091601, trajectory_id]). Both distributions use a periodic
[0,2π)² domain, viscosity0.01, forcing−4cos(4y), full-spectrum
ETDRK4, internal dt0.005, output dt0.02, N64, padding99, float32/complex64.

## Released input identity

On the experiment machine, set `GIFT_DATA_ROOT=D:/2_GIFT/canonical_data` for S3. The directory `s3_gaussian/` contains the dense and Lite observation files with cohort identity `gaussian_s3`. The released models carry the same identity and a matching frozen-generator binding. Each observation array and every learned tensor, optimizer and RNG value is identical to its original training evidence. File hashes of the current metadata-bearing files are registered separately from the original measurement checkpoint hashes. The measured results and plotting arrays are unchanged; this identity mirror is neither a fresh training run nor a regenerated population.

## Independent generation and training

Generate a new output directory, package the completed data, then set
`GIFT_DATA_ROOT` to its parent data-package directory. Data stay outside GitHub.

```shell
python -m scripts.generate_gaussian_data --output ../runs/s3_generation --device cuda --execute
python -m scripts.prepare_gaussian_package --generation ../runs/s3_generation --output ../gift-data/s3_gaussian
```

For interrupted generation, add `--resume` with the same output. Diagnostic
`--subset` or `--pilot-steps` outputs cannot qualify as the complete S3 population.
Generation's committed state includes solver state and already stored frames.

Run each model separately after downloading official repositories and preparing
the external runtime interfaces outside GIFT as documented in
[EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md). Set
`GIFT_EXTERNAL_ADAPTER_ROOT` to that independent runtime directory:

```shell
python -m scripts.run_training gift_generator --run-training --dataset ../gift-data/s3_gaussian/trajectories.h5 --output ../runs/s3_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../gift-data/s3_gaussian/trajectories.h5 --low-model ../runs/s3_generator/model.pt --seed 20260820 --output ../runs/s3_gift_20260820
python -m scripts.run_training gift_low --run-training --data-profile gaussian --output ../runs/s3_lite_generator
python -m scripts.run_training gift_branch --run-training --data-profile gaussian --low-model ../runs/s3_lite_generator/gift_main.pt --seed 20260820 --output ../runs/s3_lite_20260820
python -m scripts.run_training fno2d --run-training --data-profile gaussian --output ../runs/s3_fno2d
python -m scripts.run_training fno3d --run-training --data-profile gaussian --output ../runs/s3_fno3d
python -m scripts.run_training uno --run-training --data-profile gaussian --output ../runs/s3_uno
python -m scripts.run_training unet --run-training --data-profile gaussian --output ../runs/s3_unet
```

Repeat each GIFT branch command independently for seeds20260821 and20260822.
Full-data generator and branch each receive500epochs, reported separately.
All four baselines receive500epochs. The two distributions use the same data
size, model structure, loss, optimizer, batch size, rollout lengths, random seeds
and selection rules. GIFT-Lite uses50 trajectories,12000 generator updates and
5834 branch updates per seed. Update counts and timing scope are reported in
[EXPERIMENTS.md](../EXPERIMENTS.md) section5.3.

For full-data GIFT and the Lite generator, replace `--run-training` with
`--resume` to continue the same attempt. For Lite branches and the four baselines,
retain `--run-training` and add `--resume`. Use the same output, source, data and
runtime; a published model is not an optimizer-state continuation checkpoint.

## Evaluation and figure protocol

S3 uses the t=5 anchor, baseline46-frame context and150-step future through
t=8. All180 test trajectories are evaluated. Failed predictions, if any,
remain in numerical records with finite counts; they are never silently omitted
from a reported full-population comparison.

`python -m scripts.run_experiment S3 --output ../runs/S3` uses the published
checkpoint layout under `artifacts/s3_gaussian/`. Each model path can instead be
specified explicitly using the evaluation entry point's path overrides. Numerical evaluation and
plotting are independent; use `--skip-plots` then:

```shell
python -m experiments.formal.s3_initial_distribution.plot_results --result-dir ../runs/S3 --output-dir ../runs/S3_figures
```

The paired curve places four-vortex measurements on the left and Gaussian
measurements on the right, with common axes and method colours. The keyframe
plates use prespecified IDs1045 and2265, each the sixth trajectory of its test
set, with GIFT branch seed20260820. The experiment record embeds PNG previews
and links the original SVG files. Colour limits are fixed per distribution;
no outcome-dependent trajectory selection or method-specific scaling is used.

## Results and interpretation

The S3 numerical summaries and figures are available at
`results/formal/S3_initial_distribution/`: summary tables, the paired curve, the
Gaussian keyframe plate and their source arrays, with `published.json` binding the
measurement, executed sources and released inputs. The measured comparison and its
limitations are reported in [EXPERIMENTS.md](../EXPERIMENTS.md) section 5.3.

Every test trajectory is retained. If a method's later report times no longer
cover all 180 test trajectories because some predictions became non-finite, its
curve is stopped at the last complete time and annotated on the panel rather than
drawn from a finite subset, its per-time `finite_count` is carried in
`summary/metrics.csv`, and the endpoint summary records
`status: nonfinite_predictions` instead of a full-population mean. Do not read a
finite-subset mean as the whole-population result.

Keyframe colour limits are prespecified constants and are never fitted to the
plotted trajectory. The four-vortex plate keeps the paper's ±19 (scalar field) and
±21 (signed prediction error). The Gaussian population declares its own prespecified pair,
**±25 for both quantities**, because its reference field exceeds ±19: the
prespecified example trajectory 2265 reaches about 24.06. A prespecified range the
selected keyframe would exceed makes the renderer refuse the figure rather than
widen the scale, and the two limits are recorded in `figure_manifest.json`.
