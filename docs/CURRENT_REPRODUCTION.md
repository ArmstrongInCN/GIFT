# Reproducing the current W1 W2 and W3 protocols

The current prediction protocol uses N300/N50, 76 real observation nodes and one true initial field. ANIE uses `aligned_ns_single_v1`. W3 uses 50 parameter cases, one training trajectory and zero independent validation trajectories per case, with fixed terminal GIFT checkpoints. The following entry points prepare these protocols without the experiment machine's old run folders or cost forecasts. They never update the supplied observations, reference weights or published numerical records.

Install the environments in [SETUP.md](SETUP.md), then prepare the pinned upstream sources and external interfaces using [EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md). The external algorithm implementations remain outside GIFT. The commands below run from the repository root and use new output directories.

## Inputs and environment

```powershell
$env:GIFT_DATA_ROOT = 'D:\datasets\GIFT-dataset'
$env:GIFT_EXTERNAL_ROOT = 'D:\upstreams'
$env:GIFT_ANIE_SOURCE_ROOT = 'D:\upstreams\anie'
$env:GIFT_EXTERNAL_ADAPTER_ROOT = 'D:\gift-external-runtime'
python -B -m scripts.verify_data --root $env:GIFT_DATA_ROOT --full-array-scan
```

Use the reorganized dataset containing `dataset.json`, including roles `w1_w2.train300`, `w1_w2.train50` and `w3.cases`. The existing ScienceDB DOI identifies the data record; the 2026-10-20 package is prepared locally and has not yet been published as an updated online version. No training observations are required in Git. Existing absolute paths inside old model/data provenance describe the original measurement machine; they are not new output locations.

## Fresh W1 and W2 training

Prepare all seven fixed jobs before accessing an evaluation population:

```powershell
$Run = 'D:\runs\gift_prediction_fresh'
python -B -m scripts.reproduce_prediction prepare --root $Run --dataset-root $env:GIFT_DATA_ROOT --device cuda

# A command without --execute prints its fixed recipe and does not train.
python -B -m scripts.reproduce_prediction train --root $Run --method fno2d

# Each command trains exactly one method using its declared native budget.
python -B -m scripts.reproduce_prediction train --root $Run --method GIFT --execute
python -B -m scripts.reproduce_prediction train --root $Run --method fno2d --execute
python -B -m scripts.reproduce_prediction train --root $Run --method fno3d --execute
python -B -m scripts.reproduce_prediction train --root $Run --method anie_ns --execute
python -B -m scripts.reproduce_prediction train --root $Run --method uno --execute
python -B -m scripts.reproduce_prediction train --root $Run --method unet --execute
python -B -m scripts.reproduce_prediction train --root $Run --method GIFT-N50 --execute

python -B -m scripts.reproduce_prediction finalize --root $Run
```

Preparation reads and fingerprints the complete N300/N50 train/validation windows, verifies that N50 is the ordered first50 with identical validation, and freezes the jobs and base protocol. It uses the controller's existing configurations: GIFT generator/branch each500 epochs, FNO/U-NO/U-Net terminal epoch500, aligned ANIE's original validation-best/early-stopping recipe, and the reduced-data GIFT-N50 recipe. It does not copy or alter the observations. The source roots can also be supplied explicitly through `prepare --source-root ... --anie-root ...`.

After an infrastructure interruption, use the same method, root and `--resume --execute`. Existing slots cannot be overwritten by a new training command. Scientific failures remain their prescribed slots; create a distinct run to change the experiment. `finalize` validates all available checkpoints against the frozen recipes and retains declared scientific failures before writing the evaluation bindings. Missing slots or unresolved infrastructure interruptions prevent finalization. Do not use test results to change these recipes or select checkpoints.

Evaluate into separate new directories:

```powershell
python -B -m scripts.reproduce_prediction evaluate --root $Run --dataset-root $env:GIFT_DATA_ROOT --experiment W1 --cohort confirmation --output 'D:\evaluations\w1_confirmation' --device cuda
python -B -m scripts.reproduce_prediction evaluate --root $Run --dataset-root $env:GIFT_DATA_ROOT --experiment W2 --cohort confirmation --output 'D:\evaluations\w2_confirmation' --device cuda
```

Repeat with `--cohort old_test` and new output directories for the separately reported exploratory population. W1 evaluates all seven fixed slots on future75 nodes; W2 excludes ANIE and evaluates six slots on the three native grids and future25 nodes. Both preserve all180 trajectories and failure records. Add `--plots` to plot committed values. For an interrupted evaluation use its original output with `--resume`; completed raw arrays and metric bindings are checked before reuse. Newly trained models report their own measured values and costs, not the historical table values.

## Reference model bundle

The [reference model bundle](https://github.com/ArmstrongInCN/GIFT/releases/download/current-protocol-20261020/GIFT-current-models-20261020.zip), `GIFT-current-models-20261020.zip`, contains seven current inference models and the two paired GIFT generators. It contains no observations, upstream algorithm source or complete optimizer journals. Its archive and model-file identities are recorded in [CURRENT_MODEL_BUNDLE.json](CURRENT_MODEL_BUNDLE.json). Download it from the [current-protocol release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020) and verify the recorded SHA256 before extraction.

After extracting a supplied bundle into a new directory:

```powershell
$Bundle = 'D:\models\gift_current'
$Active = 'D:\runs\gift_current_inference'
python -B -m scripts.package_current_models verify --bundle $Bundle
python -B -m scripts.package_current_models activate --bundle $Bundle --output $Active
python -B -m scripts.reproduce_prediction evaluate --root $Active --dataset-root $env:GIFT_DATA_ROOT --experiment W1 --cohort confirmation --output 'D:\evaluations\reference_w1' --device cuda
python -B -m scripts.reproduce_prediction evaluate --root $Active --dataset-root $env:GIFT_DATA_ROOT --experiment W2 --cohort confirmation --output 'D:\evaluations\reference_w2' --device cuda
```

Activation checks every bundle member and both GIFT branch/generator tensor pairs, retains all weight bytes and scientific model specifications, and writes current-machine lock pointers separately. A copied ANIE operational receipt uses a relative best-checkpoint pointer; its original receipt is retained byte-for-byte in the bundle. GIFT resolves its sibling generator and still verifies the generator tensor binding. External model construction and forward operations use the user's current source roots without rewriting the recorded model metadata. Current distribution file hashes may differ from historical measurement hashes after earlier descriptive-metadata normalization; these byte identities remain separately recorded, and this export does not rewrite either weight file.

To prepare another reference bundle from completed source runs, use explicit roots or the external source registry:

```shell
python -B -m scripts.package_current_models export --native-root /runs/native --extension-root /runs/extension --aligned-root /runs/aligned --output /exports/gift-current-models
```

The exporter validates all fixed slots before copying the nine model files. It refuses an existing destination. Exporting or activating a bundle does not publish it, and reference inference weights cannot be used as the resume journal for an unrelated training run.

## Current W3

Follow [W3_REPRODUCTION.md](W3_REPRODUCTION.md) for the complete input preparation, five native methods, per-case measurement, current 250-slot aggregation and optional single-trajectory regeneration. `scripts.prepare_w3_run` builds the current run-local gate by validating the downloaded cases. `scripts.generate_w3_parameter_data --profile single_trajectory_terminal_v1` directly regenerates the current protocol into a new directory. The historical validation-best/three-noise workflows are separate protocols.

## Numerical comparisons and release checks

File-integrity checks require exact bytes; a numerical comparison considers dtype, absolute scale, relative norms, finite/failure masks, physical time nodes and solver conventions. Compare the same inputs and fixed weights before interpreting a difference as a code defect. CPU/GPU FFT and convolution paths can produce small float32 differences, and long integrations can amplify them. Preserve the new values and diagnostic errors instead of changing the reference observations, weights or reported results.

The release repair checks regenerated W3 case000 on CPU and on the original GPU: the GPU's 501 saved fields matched the supplied observations bit-for-bit, while CPU full-trajectory relative L2 differed by approximately 2.02e-4 (endpoint 5.15e-4). Short restarted intervals differed only at about 1e-7. This identifies a cross-device numerical difference for that checked case, not a blanket tolerance for every result. A fresh native U-Net continuation also passed; a separate earlier CPU run had approximately 2.15e-7 maximum parameter difference with unchanged RNG, optimizer step and losses. Its test uses both a tensor-scale bound and relative-norm bound for floating arrays, while integer state, progress, RNG, normalization and failure masks remain exact.

```shell
python -B -m pytest -q -p no:cacheprovider
python -B -m scripts.audit_repository
```

Native CPU continuation tests are opt-in through a new `GIFT_BASELINE_TEST_ROOT`; TensorFlow and CUDA tests use their documented separate environments. Skips are reported separately. Result-package and dataset hash verification do not replace full-budget training or inference. After staging the intended local file selection, audit that selection again before a later upload.
