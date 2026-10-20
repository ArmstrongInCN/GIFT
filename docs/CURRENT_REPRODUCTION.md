# Reproducing the current W1 W2 and W3 protocols

The current prediction protocol uses N300/N50, 76 real observation nodes and one true initial field. ANIE uses `aligned_ns_single_v1`. W3 uses 50 parameter cases, one training trajectory and zero independent validation trajectories per case, with fixed terminal GIFT checkpoints. The following entry points prepare these protocols without the experiment machine's old run folders or cost forecasts. They never update the supplied observations, reference weights or published numerical records.

Install the environments in [SETUP.md](SETUP.md), then prepare the pinned upstream sources and external interfaces using [EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md). The external algorithm implementations remain outside GIFT. Source availability and recorded hashes do not grant a licence: the [licence table](../README.md#data-code-and-licence) includes ANIE's unresolved authorization status. The commands below run from the repository root and use new output directories.

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

The [reference bundle](https://github.com/ArmstrongInCN/GIFT/releases/download/current-protocol-20261020-source-free/GIFT-current-models-20261020-source-free.zip), `GIFT-current-models-20261020-source-free.zip`, contains seven current inference models and the two paired GIFT generators. It is available from the [source-free release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020-source-free). Its archive and model-file identities are recorded in [CURRENT_MODEL_BUNDLE.json](CURRENT_MODEL_BUNDLE.json). It contains no observations or complete optimizer journals, and its checkpoint metadata are checked for source snippets as well as executable payloads. Verify the recorded SHA256 before extraction.

The earlier `GIFT-current-models-20261020.zip` at the [superseded release](https://github.com/ArmstrongInCN/GIFT/releases/tag/current-protocol-20261020) contains four ANIE `before`/`after` source strings (441 characters) in checkpoint metadata. It remains available for provenance and must not be described as containing no upstream source. The new release supersedes that bundle with a separate publication copy that removes those strings and verifies every tensor exactly against its original. The original weights remain read-only; metadata cleanup changes the ANIE file's serialized bytes and hash. It neither retrains the model nor resolves ANIE's outstanding authorization.

After extracting a supplied bundle into a new directory:

```powershell
$Bundle = 'D:\models\gift_current'
$Active = 'D:\runs\gift_current_inference'
python -B -m scripts.package_current_models verify --bundle $Bundle
python -B -m scripts.package_current_models activate --bundle $Bundle --output $Active
python -B -m scripts.reproduce_prediction evaluate --root $Active --dataset-root $env:GIFT_DATA_ROOT --experiment W1 --cohort confirmation --output 'D:\evaluations\reference_w1' --device cuda
python -B -m scripts.reproduce_prediction evaluate --root $Active --dataset-root $env:GIFT_DATA_ROOT --experiment W2 --cohort confirmation --output 'D:\evaluations\reference_w2' --device cuda
```

Activation checks every bundle member and both GIFT branch/generator tensor pairs, retains the verified publication-copy bytes and scientific model specifications, and writes current-machine lock pointers separately. A copied ANIE operational receipt uses a relative best-checkpoint pointer; a separately bound source-receipt snapshot is also checked for source metadata during export. GIFT resolves its sibling generator and still verifies the generator tensor binding. External model construction and forward operations use the user's current source roots. The original measurement-file hashes, current source-file hashes and publication-copy hashes are distinct provenance identities. The exporter keeps originals unchanged and records the publication-copy identity separately; activation does not serialize the weights again.

To prepare another reference bundle from completed source runs, use explicit roots or the external source registry:

```shell
python -B -m scripts.package_current_models export --native-root /runs/native --extension-root /runs/extension --aligned-root /runs/aligned --output /exports/gift-current-models
```

The exporter validates all fixed slots, creates the nine publication files and checks their metadata and exact tensor equivalence. It refuses an existing destination. Exporting or activating a bundle does not publish it, and reference inference weights cannot be used as the resume journal for an unrelated training run.

## Current W3

Follow [W3_REPRODUCTION.md](W3_REPRODUCTION.md) for the complete input preparation, five native methods, per-case measurement, current 250-slot aggregation and optional single-trajectory regeneration. `scripts.prepare_w3_run` builds the current run-local gate by validating the downloaded cases. `scripts.generate_w3_parameter_data --profile single_trajectory_terminal_v1` directly regenerates the current protocol into a new directory. The historical validation-best/three-noise workflows are separate protocols.

## Numerical comparisons and release checks

File-integrity checks require exact bytes; a numerical comparison considers dtype, absolute scale, relative norms, finite/failure masks, physical time nodes and solver conventions. Compare the same inputs and fixed weights before interpreting a difference as a code defect. CPU/GPU FFT and convolution paths can produce small float32 differences, and long integrations can amplify them. Preserve the new values and diagnostic errors instead of changing the reference observations, weights or reported results.

The release repair checks regenerated W3 case000 on CPU and on the original GPU: the GPU's 501 saved fields matched the supplied observations bit-for-bit, while CPU full-trajectory relative L2 differed by approximately 2.02e-4 (endpoint 5.15e-4). Short restarted intervals differed only at about 1e-7. This identifies a cross-device numerical difference for that checked case, not a blanket tolerance for every result. A fresh native U-Net continuation also passed; a separate earlier CPU run had approximately 2.15e-7 maximum parameter difference with unchanged RNG, optimizer step and losses. Its test uses both a tensor-scale bound and relative-norm bound for floating arrays, while integer state, progress, RNG, normalization and failure masks remain exact.

```shell
python -B -m pytest -q -p no:cacheprovider
python -B -m scripts.audit_repository
```

Native CPU continuation tests are opt-in through a new `GIFT_BASELINE_TEST_ROOT`; TensorFlow and CUDA tests use their documented separate environments. Skips are reported separately. Result-package and dataset hash verification do not replace full-budget training or inference. Audit the intended publication selection and attachment after any change. Current-file and model-bundle checks do not certify Git history: old adapters remain in earlier `main` commits and the `reference-checkpoints-20260913` tag. This release provides the replacement attachment without rewriting Git history or removing the superseded attachment; see [EXTERNAL_REPOSITORY_POLICY.md](EXTERNAL_REPOSITORY_POLICY.md) and the final checks in [VALIDATION.md](VALIDATION.md).
