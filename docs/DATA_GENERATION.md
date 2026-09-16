# Independent data generation / 独立数据生成

**Training from scratch on supplied fields** and **generating those fields again**
are different workflows. Fresh training initializes a model without trained
weights. Regeneration additionally integrates the PDE from specified initial
conditions and creates the observation inputs. Neither is achieved by renaming
or relabelling a downloaded checkpoint.

从零训练可使用单独下载的数据包；从初值重新生成仿真场还涉及求解器及浮点环境。
模型训练、数据生成和已有检查点的快速读出分别使用独立命令。

## Environment and boundaries

Use the prediction/GIFT environment in [SETUP.md](SETUP.md). The project's
`src/even_full_spectrum_ns.py` implements full-spectrum Fourier pseudospectral
ETDRK4, internal step 0.005, float32/complex64. Physical parameters, grid sizes,
source hashes and numerical runtime are recorded in each generation attempt.

Run commands from the project root. Each `--output` must be a **new directory
outside both the repository and `GIFT_DATA_ROOT`**. Omit `--execute` for a
read-only plan. These are separate commands, not an all-experiment program.

## Clean fields

```shell
python -m scripts.generate_data --dataset standard --device cuda --output ../generation/standard --execute
python -m scripts.generate_data --dataset fno-training --initial-conditions ../GIFT-data/initial_conditions/baseline_extra950.json --device cuda --output ../generation/fno-training --execute
python -m scripts.generate_data --dataset fno-test --device cuda --output ../generation/fno-test --execute
```

The solver's standard container stores 50 training, 20 validation and 200
evaluation trajectories. The canonical export below places 20 of the evaluation
trajectories in validation and keeps 180 independent test trajectories.
Common prediction training uses the 50 training initial conditions plus 950
explicit parameter sets, public IDs 50–999. The generation routine translates
these IDs to its internal storage identities without changing parameters.
The extra-950 JSON contains parameters,
not saved vorticity; its original random-draw seed is unspecified. The provided
parameter values suffice to integrate these trajectories without saved truth.

Keep default batch sizes and the recorded environment when reproducing a
generation profile. `--subset`, `--split` selections and `--pilot-steps` are
partial/diagnostic options, not evidence for a full-population experiment.

### Exact coarse frames

After the corresponding dense parent finishes:

```shell
python -m scripts.derive_dense_frames --dataset fno-training-coarse --parent ../generation/fno-training --output ../generation/fno-training-coarse --execute
python -m scripts.derive_dense_frames --dataset short-test --parent ../generation/fno-test --output ../generation/short-test --execute
python -m scripts.derive_dense_frames --dataset cross-resolution --parent ../generation/fno-test --output ../generation/cross-resolution --execute
```

Derivation selects exact integer frames without interpolation or another solver
invocation. It requires a completed new parent attempt, not a downloaded HDF5
substituted as that parent. Its receipt says `COMPLETE_DERIVATION`. Independent
integration of these three dataset names is also supported by `generate_data`;
choose one approach per collection slot.

## Noise and PINN sampling

```shell
python -m scripts.generate_noise --clean ../generation/standard/data.h5 --output ../generation/noise --execute
python -m scripts.generate_sampling --input ../generation/standard/data.h5 --output ../generation/sampling-clean --execute
python -m scripts.generate_sampling --input ../generation/noise/noise_001.h5 --output ../generation/sampling-001 --execute
python -m scripts.generate_sampling --input ../generation/noise/noise_010.h5 --output ../generation/sampling-010 --execute
```

Noise uses paired MT19937 seed-0 standard normals at 1% and 10% of each clean
group's standard deviation, without finite-sample RMS renormalization. Sampling
uses seed 1234, trajectory 0, 500 sensors, 60 times, a 24,000/6,000 observation-row
split and 60,000 LHS physics points. No neural-network weights are exported.

## Same-attempt continuation / 同次运行断点续算

Add `--resume` to the same command with unchanged output, source, inputs and
numerical environment. Integration saves completed solver-step boundaries;
noise and derivation save completed block boundaries. Sampling saves its random
design before generating targets. Uncommitted work may repeat after interruption.
Completed outputs are not overwritten, and another run's checkpoint is rejected.

`--stop-after-steps`, `--stop-after-chunks` and `--stop-after-design` provide
operational pauses for their respective programs. They do not reduce the final
declared dataset or certify a reduced attempt as a full result.

## Assemble completed inputs

Assembly verifies and packages independent completed jobs; it does not train
models or fill gaps using downloaded truth. In PowerShell:

```powershell
python -m scripts.assemble_generated_data `
  --job standard=../generation/standard `
  --job fno-training=../generation/fno-training `
  --job fno-test=../generation/fno-test `
  --job fno-training-coarse=../generation/fno-training-coarse `
  --job short-test=../generation/short-test `
  --job cross-resolution=../generation/cross-resolution `
  --job noise=../generation/noise `
  --job sampling-clean=../generation/sampling-clean `
  --job sampling-001=../generation/sampling-001 `
  --job sampling-010=../generation/sampling-010 `
  --output ../generated-inputs --execute
```

For bash, replace the line-continuation backticks with backslashes. Assembly
checks source/receipt identities, SHA-256, IDs, times, initial parameters, numeric
shapes/types and finite values. Preserve its provenance, manifest and splits
together. Successful assembly establishes input integrity, not equality of
downstream experiment results. This generated collection has a different schema
from the downloadable package checked by `scripts.verify_data`.

## Canonical prediction package

After all jobs have been assembled, export the publication numbering:

```shell
python -m scripts.prepare_prediction_package --source ../generated-inputs --output ../generated-prediction-data
python -m scripts.verify_data --root ../generated-prediction-data --full-array-scan
```

The new package includes README, dictionary, splits, manifest and CC BY 4.0
notice. Prediction IDs are training 0–999, validation 1000–1039 and test
1040–1219. All resolutions share the same IDs. Observation arrays are copied
losslessly; the M1 files retain their separate local IDs and unchanged bytes.
Generation receipts are preserved. The command rejects incomplete collections,
existing output directories and already-canonical inputs.

## Train on regenerated fields

For prediction, set `GIFT_DATA_ROOT` to `../generated-prediction-data` and use
`--data-profile canonical` for an independent FNO-2D, FNO-3D, U-NO or U-Net
training command. Full-data GIFT's `gift_generator` and `gift_predictor` commands
take the explicit canonical dense `--dataset` path instead. Its branch requires
the completed full-data generator trained on the same observations; downloaded
or reduced-data generator weights do not qualify as its fresh prerequisite.

For M1 or GIFT-Lite regeneration, retain `GIFT_DATA_ROOT=../generated-inputs` and
use the `--data-profile regenerated` route. These consumers keep the original
identification/small-data protocol and local IDs.

For native PINN training, switch to the separate PINN environment, retain the
same generated `GIFT_DATA_ROOT`, and choose one condition and library:

```shell
python -m scripts.run_training pinn --execute --data-profile regenerated --mode known --condition noise_000 --output ../runs/generated_pinn/noise_000/known
```

Use `--mode open` for the open library and select `noise_001` or `noise_010`
separately as needed. Add `--resume` to continue that same output directory.
The input gate verifies the clean/noise/sampling provenance chain, hashes,
population and split metadata, and recomputes observation targets from the
declared HDF5 input. The network, loss and native optimizer are unchanged.
The default `--data-profile released` continues to require the supplied data's
exact hashes. Never bypass either profile's checks or mix their checkpoints.

Successful input checks and small native continuation tests establish the
execution route, not agreement of full-budget trained metrics. Regenerated
inputs retain their own source and numerical identities in each training run.

跨设备不承诺逐比特一致。检验时分别记录文件哈希、数组差异、训练权重和实验指标，
不能将其中一种核验替代其它层次的验证。

## S4 Gaussian initial-condition population

The independent [S4 protocol](S4_PROTOCOL.md) specifies a smooth Gaussian random
field and IDs1220–2439, split before training. Generate it with
`scripts.generate_gaussian_data` and package the completed run with
`scripts.prepare_gaussian_package`. Neither command reuses model checkpoints.
To append the verified `s4_gaussian` directory to an existing canonical data
package, preserve a separate metadata backup:

```shell
python -m scripts.append_gaussian_package --package ../staging/s4_gaussian --dataset ../gift-data --metadata-backup ../data-metadata-backup
```

This explicit append copies only new files, updates the root dictionary, splits,
schema and manifest, and verifies every numeric array. It does not alter any
existing scientific input. Both destination subdirectory and backup must be new;
the original license remains CC BY4.0. S4 uses `--data-profile gaussian` for the
four baselines and GIFT-Lite; full-data GIFT takes the explicit Gaussian dataset.
