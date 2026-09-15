# Independent experiments / 独立实验运行

Use the prediction/GIFT environment and external roots from [SETUP.md](SETUP.md).
Each command below runs **one experiment**, reads selected trained weights and
writes to a new output directory. It does not train all models. The optional
`scripts.run_experiment` wrapper configures only its child process: two CPU
threads for prediction/GIFT inference, one for M1 regression/coefficient readout,
one visible GPU, and no global TF32 override. Place `--device cpu` before the
experiment name for an explicitly recorded CPU evaluation.

以下命令彼此独立。使用随项目提供的完整权重可跳过训练；使用自己从零训练得到的
权重时，显式指定相应路径。输出目录不能与数据、源码或已有结果重叠。

## Evaluate the currently completed prediction models

The supplied prediction tables use GIFT-Lite and the applicable baselines.
Full-data GIFT branches are still training. These commands do not require them:

```shell
python -m scripts.run_experiment M2 --gift-regimes GIFT-Lite --output ../runs/M2_lite --skip-plots
python -m scripts.run_experiment M3 --gift-regimes GIFT-Lite --output ../runs/M3_lite --skip-plots
python -m scripts.run_experiment S1 --gift-regime GIFT-Lite --output ../runs/S1_lite
python -m scripts.run_experiment S2 --gift-regimes GIFT-Lite --output ../runs/S2_lite
python -m scripts.run_experiment S3 --gift-regime GIFT-Lite --output ../runs/S3_lite
```

`--gift-regimes` explicitly selects completed regimes for M2/M3/S2; omitting it
requires both GIFT and GIFT-Lite. S1/S3 run one `--gift-regime` per command.
Selection never silently skips a missing requested model or a failed trajectory.
当前已完成结果可直接重算，不必等待全数据 GIFT；这些命令只做评价，不启动训练。

## M1: equation identification

Run one method and one condition per command. The packaged GIFT/PINN defaults
are selected by condition; PDE-FIND computes regression directly from observations.

```shell
python -m scripts.run_experiment M1 --execute --method GIFT --condition noise_000 --output ../runs/m1_jobs
python -m scripts.run_experiment M1 --execute --method PDE-FIND --condition noise_000 --output ../runs/m1_jobs
python -m scripts.run_experiment M1 --execute --method PDE-FIND-KC --condition noise_000 --output ../runs/m1_jobs
python -m scripts.run_experiment M1 --execute --method PINN-SR --condition noise_000 --output ../runs/m1_jobs
python -m scripts.run_experiment M1 --execute --method PINN-SR-KC --condition noise_000 --output ../runs/m1_jobs
```

Repeat each independently for `noise_001` and `noise_010`. After all 15 jobs finish:

```shell
python -m experiments.formal.m1_equation_identification.aggregate --jobs ../runs/m1_jobs --output ../runs/M1 --skip-plots
```

For your own trained GIFT model, add `--checkpoint /path/to/gift_main.pt` and
`--checkpoint-sha256 SHA256`. For PINN, the checkpoint argument is its completed
`result.json`, kept beside `terminal_state.npz` and `COMPLETED.json`; hash that
JSON, not a different member. For example, PowerShell can calculate the value:

```powershell
$pinnResult = (Resolve-Path '../runs/pinn/noise_000/known/result.json').Path
$pinnHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $pinnResult).Hash
python -m scripts.run_experiment M1 --execute --method PINN-SR-KC --condition noise_000 --checkpoint $pinnResult --checkpoint-sha256 $pinnHash --output ../runs/my_m1_jobs
```

Replace `--execute` with `--use-results` to verify and read a previously committed
M1 job without checkpoint inference or regression. A partial job is not a result.

## M2 and M3: prediction

```shell
python -m scripts.run_experiment M2 --output ../runs/M2 --skip-plots
python -m scripts.run_experiment M3 --output ../runs/M3 --skip-plots
```

M2 evaluates GIFT, GIFT-Lite and four baselines on 180 independent test trajectories
at N64, t = 5–8. M3 evaluates both GIFT regimes and the two FNO models on the same
180 paired trajectories at N64/N96/N128, t = 5–6. Public test IDs are 1040–1219.
Both use all three declared seeds for each GIFT regime, not one selected seed. These are actual
forward computations from observations and weights, not copies of report tables.

To use newly trained weights, both commands accept `--low-model`, `--fno2d-model`
and `--fno3d-model`. M2 additionally accepts `--uno-model` (ordinary `.pt` or a
split `weights.json`) and `--unet-model`. Supply all three GIFT branches using:

```shell
--gift-model 20260820=/path/to/branch20/model.pt --gift-model 20260821=/path/to/branch21/model.pt --gift-model 20260822=/path/to/branch22/model.pt
```

The paths must point to the actual completed model exports. The low generator
must be the prerequisite used to train these branches. Do not mix prerequisites,
noise conditions, reduced diagnostic budgets or unrelated terminal weights.
GIFT-Lite defaults to the supplied reduced-data weights. To replace them with
your own reduced-data training results, use `--lite-low-model` and three repeated
`--lite-gift-model SEED=PATH` arguments. Full-data defaults live in
`artifacts/gift_full/`; set `GIFT_CHECKPOINT_ROOT` to use a different checkpoint root.

## S1, S2 and S3: ablations and seed stability

```shell
python -m scripts.run_experiment S1 --gift-regime GIFT --output ../runs/S1_GIFT
python -m scripts.run_experiment S1 --gift-regime GIFT-Lite --output ../runs/S1_GIFT_Lite
python -m scripts.combine_regime_results --experiment S1 --gift ../runs/S1_GIFT --gift-lite ../runs/S1_GIFT_Lite --output ../runs/S1
python -m scripts.run_experiment S2 --output ../runs/S2
python -m scripts.run_experiment S3 --gift-regime GIFT --output ../runs/S3_GIFT
python -m scripts.run_experiment S3 --gift-regime GIFT-Lite --output ../runs/S3_GIFT_Lite
python -m scripts.combine_regime_results --experiment S3 --gift ../runs/S3_GIFT --gift-lite ../runs/S3_GIFT_Lite --output ../runs/S3
```

S1 disables the high-frequency branch of each selected GIFT model; S2 compares
recursive local correction on/off; S3 reads and evaluates all three trained
seeds. These commands do not start branch retraining. All accept `--low-model`
and the same three `--gift-model SEED=PATH` arguments, plus the GIFT-Lite overrides
above. S1/S3 retain independently resumable runs for each training regime;
the combination command verifies both completed reports and joins their tables
without retraining or rerunning inference. S2 evaluates both regimes on one
common independent test population. It preserves failures and
reports finite-only summaries separately; a finite-only mean is not a mean over
the full population. No training trajectories are used as supplementary tests.

## Recovery, numerical outputs and SVGs

M2/M3/S1/S2/S3 save checksummed numerical calls at method/seed/grid/cohort
boundaries. Add `--resume` to the identical command with its original output
directory. Completed calls are verified and reused; an interrupted call repeats.
Changed source, data, weights or numerical runtime are rejected. Resumption
prints the new aggregation directory inside the original run; use that directory
as the input to figure rendering. It does not overwrite the prior aggregation.

`summary/metrics.csv` contains summary values; `raw/` preserves arrays or fields.
Keep the completion record and file hashes together. M2/M3 can plot after numeric
completion when `--skip-plots` is omitted. To redraw the five report SVGs separately,
follow [FIGURES.md](FIGURES.md). A figure failure does not invalidate already
committed numerical outputs and does not require retraining.

Check one complete numerical output and, optionally, its separately rendered figures:

The compact results included in the repository are described in [RESULTS.md](RESULTS.md).
They have a separate integrity checker; they are not full experiment directories.

```shell
python -m scripts.verify_results --experiment M2 --result-dir ../runs/M2 --figures ../runs/figures_M2_curve --figures ../runs/figures_M2_keyframes
```

This read-only check verifies completion, file hashes, SVG structure and the
figure-to-report binding. It does not recompute metrics, inspect visual layout or
establish scientific accuracy; those checks remain separate.

数值结果、表格和 SVG 必须来自同一组已完成实验。保留非有限预测和失败记录，
不通过换检查点、删轨迹或改色限来维持预期的模型排名。
