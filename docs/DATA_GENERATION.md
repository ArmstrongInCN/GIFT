# Independent data generation / 独立数据生成

当前W3单轨迹数据使用 `scripts.generate_w3_parameter_data --profile single_trajectory_terminal_v1` 直接生成；完整准备、采样和新run gate步骤见 [W3_REPRODUCTION.md](W3_REPRODUCTION.md)。默认 `validation_best_v1` 保留旧21轨迹协议，不能替代当前50组单轨迹实验。W1/W2从现成正式包直接训练的入口见 [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md)。

**Training from scratch on supplied fields** and **generating those fields again**
are different workflows. Fresh training initializes a model without trained
weights. Regeneration additionally integrates the PDE from specified initial
conditions and creates the observation inputs. Neither is achieved by renaming
or relabelling a downloaded checkpoint.

从零训练可使用单独下载的数据包；从初值重新生成仿真场还涉及求解器及浮点环境。
模型训练、数据生成和已有检查点的快速读出分别使用独立命令。

当前本地发布包为 `D:/GIFT/GIFT-dataset`，目录和字段见[包说明](data_package/README.md)及[数据字典](data_package/DATA_DICTIONARY.md)。2026年10月20日的重整仅复制、归类已有输入和正式数值源数据，没有重新积分或训练，也尚未更新 ScienceDB。以下再生成命令各自产生新的运行产物，不自动重建含六个实验结果的完整发布包。

## W1/W2 观测选择与配对人口

W1/W2主训练从N64数据池IDs0–999按 `default_rng(2026100304).permutation(1000)[:300]` 固定选择300条，validation为IDs1000–1019，共20条；GIFT-Lite为该有序名单前50条。新版包中的父数据为 `prediction/four_vortex/train1000_n64_t0_t10_dt0p02.h5`。t5–8每0.04秒的76个真实点，精确取dense dt0.02数据的帧250,252,…,400，不做时间插值。推理输入仅各轨迹自己的t5场；loss覆盖按各算法配方声明，U-NO/U-Net只监督20/4步。

新版直接提供 `prediction/w1_w2/train300_validation20.h5` 和 `train50_validation20.h5`。原运行的 N300 文件为 `prediction_native/data/train300_validation20_5to8_dt0p04.h5`；测量时输入manifest SHA-256为 `57c61d26b8b48e44c9518e320c2bcf97529919756d95ba1d3d2b58d662eb823f`，原绑定保存在仓库外来源登记。完整ID与源帧绑定记录在原 `PROTOCOL_LOCK.json`，模型/数据绑定文件为 `prediction_extension/EXTENSION_LOCK.json`；新版将这些来源记录复制到 `metadata/provenance/w1_w2/`，不改写历史绝对路径或科学证据。

确认IDs100000–100179和探索IDs1040–1219各180条，与training/validation不重叠。确认集采用seed2026100302记录初值参数，但其中四种方法成绩此前已被查看，不能视为全部方法盲测；探索人口另行评价。W2同ID使用逐元素相同float64初值参数，在N64/N96/N128各自网格积分真实真值；N96/N128使用ETDRK4内部dt0.005、padding147/195，存储dense dt0.02，再精取t5–6每0.04秒26节点。高网格真值不以N64插值替代。ANIE仅参加W1。

新版确认真值为 `prediction/w1_w2/confirmation_n64.h5`（t5–8、dt0.02、151帧）和 `confirmation_n96_n128.h5`（t5–6、dt0.02、51帧），正式评价仍按整数帧取76/26节点。探索人口的dense观测位于 `prediction/four_vortex/test_n64_n96_n128_dt0p02.h5`。

W1主指标排除输入初态，评价future75；W2评价future25。GIFT内部RK4 dt0.02、保存dt0.04和监督时距0.08/0.48秒属于模型积分与训练配置，不改变上述真实数据节点。W3局部识别和S1–S3独立N1000/GIFT-Lite协议采用下列各自数据生成与编号定义，不能把其样本量或46帧基线历史输入解释为W1/W2设置。

## Environment and boundaries

Use the prediction/GIFT environment in [SETUP.md](SETUP.md). The project's
`src/even_full_spectrum_ns.py` implements full-spectrum Fourier pseudospectral
ETDRK4, internal step 0.005, float32/complex64. Physical parameters, grid sizes,
source hashes and numerical runtime are recorded in each generation attempt.

Run commands from the project root. Each `--output` must be a **new directory**,
such as `runs/generation`, that does not overwrite code, input data or published
results. Omit `--execute` for a
read-only plan. These are separate commands, not an all-experiment program.

## Clean fields

The initial-condition path below uses this machine's `D:/GIFT/GIFT-dataset`;
replace that root on another machine.

```shell
python -m scripts.generate_data --dataset standard --device cuda --output ../generation/standard --execute
python -m scripts.generate_data --dataset fno-training --initial-conditions D:/GIFT/GIFT-dataset/metadata/initial_conditions_extra950.json --device cuda --output ../generation/fno-training --execute
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

## Historical noise processing and PINN sampling

Noise generation is an optional data-processing utility. W3 itself uses
50 noise-free parameter cases, one ID0 trajectory per case, no external
validation trajectory and fixed terminal GIFT checkpoints. Its case-specific
input and sampling identities are documented in
[W3_PROTOCOL.md](W3_PROTOCOL.md) and the
[data dictionary](data_package/DATA_DICTIONARY.md).

The current package provides those inputs under `identification/w3/`, including
each case's `auxiliary/pinn_sampling/noise_000_seed1234.npz`. The filename retains
the original clean-condition label; it is not the old three-noise-level W3
protocol. Historical 1%/5% inputs are absent from the new package and remain in
the separate `GIFT-dataset_legacy_20261020` backup. The commands below illustrate
independent historical/regenerated inputs, not the current 50-case campaign.

```shell
python -m scripts.generate_noise --clean ../generation/standard/data.h5 --output ../generation/noise --execute
python -m scripts.generate_sampling --input ../generation/standard/data.h5 --output ../generation/sampling-clean --execute
python -m scripts.generate_sampling --input ../generation/noise/noise_001.h5 --output ../generation/sampling-001 --execute
python -m scripts.generate_sampling --input ../generation/noise/noise_005.h5 --output ../generation/sampling-005 --execute
```

Noise uses paired MT19937 seed-0 standard normals at 1% and 5% of each clean
group's standard deviation, without finite-sample RMS renormalization. Sampling
uses seed 1234, trajectory 0, 500 sensors, 60 times, a 24,000/6,000 observation-row
split and 60,000 LHS physics points. No neural-network weights are exported.

## Same-attempt continuation / 同次运行断点续算

Add `--resume` with the same output directory, scientific inputs and integration
configuration. Source and runtime records remain descriptive; changes to them
do not block continuation. Integration saves completed solver-step boundaries;
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
  --job sampling-005=../generation/sampling-005 `
  --output ../generated-inputs --execute
```

For bash, replace the line-continuation backticks with backslashes. Assembly
checks completed receipts, data integrity, IDs, times, initial parameters, numeric
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
losslessly; the W3 files retain their separate local IDs and unchanged bytes.
Generation receipts are preserved. The command rejects incomplete collections,
existing output directories and already-canonical inputs.

This exporter retains its generated canonical layout. It does not add the
current W1/W2 training slices and confirmation fields, 50-case W3 inputs, S3 or
the six-experiment `source_data/` tree by itself. The reorganized local release
uses `dataset.json` to map roles to `prediction/four_vortex/` and the other
documented directories; keep these two packaging workflows distinct.

## Train on regenerated fields

For prediction, set `GIFT_DATA_ROOT` to `../generated-prediction-data` and use
`--data-profile canonical` for an independent FNO-2D, FNO-3D, U-NO or U-Net
training command. Full-data GIFT's `gift_generator` and `gift_predictor` commands
take the explicit canonical dense `--dataset` path instead. Its branch requires
the completed full-data generator trained on the same observations; downloaded
or reduced-data generator weights do not qualify as its fresh prerequisite.

For GIFT-Lite regeneration, retain `GIFT_DATA_ROOT=../generated-inputs` and
use the `--data-profile regenerated` route. This consumer keeps its declared
small-data protocol and local IDs. W3 separately uses the frozen ν,β,γ
50-case manifest and the matching single-ID0 data and sampling receipt for
each case; see [W3_PROTOCOL.md](W3_PROTOCOL.md). No noise is added to W3.

The following native PINN command illustrates the general regenerated-input
interface. It requires the separate PINN environment and the selected
generated `GIFT_DATA_ROOT`; choose the input condition and coefficient library:

```shell
python -m scripts.run_training pinn --execute --data-profile regenerated --mode known --condition noise_000 --output ../runs/generated_pinn/noise_000/known
```

Use `--mode open` for the open library and select `noise_001` or `noise_005`
separately as needed. Add `--resume` to continue that same output directory.
The optional package verifier checks the clean/noise/sampling provenance chain, hashes,
population and split metadata, and recomputes observation targets from the
declared HDF5 input. The network, loss and native optimizer are unchanged.
Ordinary training reads the selected HDF5/NPZ schema, population and numerical
arrays directly. Released and regenerated profiles describe the input source;
neither requires the current checkout to match its historical source bytes.
Resume checks the actual selected data and training state.

Successful input checks and small native continuation tests establish the
execution route, not agreement of full-budget trained metrics. Regenerated
inputs retain their own source and numerical identities in each training run.

跨设备不承诺逐比特一致。检验时分别记录文件哈希、数组差异、训练权重和实验指标，
不能将其中一种核验替代其它层次的验证。

## S3 Gaussian initial-condition population

The independent [S3 protocol](S3_PROTOCOL.md) specifies a smooth Gaussian random
field and IDs1220–2439, split before training. Generate it with
`scripts.generate_gaussian_data` and package the completed run with
`scripts.prepare_gaussian_package`. Neither command reuses model checkpoints.
To append the verified `s3_gaussian` directory to an existing canonical data
package, preserve a separate metadata backup:

```shell
python -m scripts.append_gaussian_package --package ../staging/s3_gaussian --dataset ../generated-prediction-data --metadata-backup ../data-metadata-backup
```

This explicit append copies only new files, updates the root dictionary, splits,
schema and manifest, and verifies every numeric array. It does not alter any
existing scientific input. Both destination subdirectory and backup must be new;
the original license remains CC BY4.0. S3 uses `--data-profile gaussian` for the
four baselines and GIFT-Lite; full-data GIFT takes the explicit Gaussian dataset.

The current local release already includes `s3_gaussian/` with the
`gaussian_s3` identity and unchanged observation arrays. Its root inventory is
managed by the reorganized package, so the historical append example above
should only target a new independently generated canonical collection, not the
current publication package.
