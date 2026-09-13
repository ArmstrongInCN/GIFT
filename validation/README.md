# Verification evidence / 验证证据

The independent [full 10% noise PINN-KC run](fresh_20260910/pinn_known_noise_010/README.md)
completed the full 31,000-update/six-STRidge protocol in 3,637.043 s. All 12 original
scalar comparisons passed and all 82 own-journal/export tensors matched. This
does not remove noise001's two failures or establish open-mode/all-M1 acceptance.
10%噪声KC完整训练数值通过，不代表其他条件或全项目通过。

The **newly integrated** standard parent also now has independently verified
[full paired noise](new_standard_noise_20260913/README.md) and
[all three sampling conditions](new_standard_sampling_20260913/README.md).
All 287,293,440 noisy values matched independent formula replay; each sampling
job's ten arrays and own design/RNG state passed independent replay. Sampling
designs match the archived designs, but targets differ (relative L2 about 0.28%)
because the new clean fields differ. These are not copies of released fields,
new model training, full collection assembly or original-number acceptance.
新清洁输入的噪声与采样派生已验证；清洁场差异及其传播仍如实保留。

The [real standard subcollection assembly](new_standard_collection_20260913/README.md)
then passed all 17 manifest-file hashes and all three full `regenerated` GIFT
input gates in 25.078 s, without mocks, tiny flags or training. Five FNO/short/cross
data slots remain explicitly absent: this is not a complete six-experiment
collection or downstream numerical acceptance. 新standard子集合已实际接通输入门禁，
不是六实验全数据或模型重训通过。

The independent [full clean PINN-KC run](fresh_20260910/pinn_known_noise_000/README.md)
completed all 31,000 NAdam updates, six STRidge rounds and the original L-BFGS
limits in 3,388.355 s. Its 82 own-journal/export state tensors agree exactly,
and **12/12 values** pass the original clean three-row comparison. This is
single-condition tolerance-level acceptance, not bit-identical reference
coefficients or full M1/open-mode acceptance. 无噪声KC完整训练的三行数值通过，
不替代其余条件；先前1%噪声的两项失败仍保留。

The [full standard clean generation](full_standard_20260913/README.md) completed
all 270 trajectories from seeds/explicit initial conditions with actual own-run
pause/resume, without reading reference fields. Integrity passed over all
37,270 frames / 152,657,920 float32 values, but field comparison is **DIFFERENT**:
aggregate relative L2 0.001559317, maximum absolute difference 0.802553177.
No whole-field tolerance, first arithmetic cause or downstream acceptance is
inferred. 完整标准数据生成与自身续算已完成，但数值不逐位相同；完整性通过不等于
实验验收，下载数据和原报告未修改。

The [full paired-noise check](full_noise_20260913/README.md) regenerated both
conditions from the released clean input with actual same-attempt pause/resume.
All 287,293,440 float32 values match the reference bitwise; independent formula,
RNG and full-group-statistic checks passed. No clean reintegration, original-input
replacement or model acceptance is inferred. 完整噪声生成及自身续算已通过逐位核验，
不等于清洁流场重生成或完整训练验收；历史失败仍保留。

The latest [runtime-guard regression](runtime_guards_20260913/README.md) covers
the new U-Net pre-import environment gate and shared checkpoint thread identity:
173 passed, 25 skipped, 3 warnings in 27.99 s (24 separate subtests). An actual
formal U-Net dry plan also passed without training or full data hashing. This
engineering result is source-hash-bound; earlier full-training and numerical
records retain their original source snapshots and unresolved failures.
最新门禁/续算工程测试与历史训练、数值验收分开；新源码不自动兼容旧任务日志，
旧任务继续使用其绑定的原源码，不改写历史证据。

`quick_20260910/` contains newly calculated summary tables, comparisons with the
unchanged archived experiment tables, and source/input/runtime bindings.
M2/M3/S1/S2/S3 use **checkpoint inference**. M1 separately combines GIFT parameter
readout, observed-data PDE regression and recovered PINN coefficient readout;
these are not fresh neural-training results. The original reference remains
under `results/formal/`.

`quick_20260910/` 保存新计算的汇总表、与原表的数值比较及源码/输入/环境记录。
M2/M3/S1/S2/S3 是检查点推理；M1 区分 GIFT 参数读出、观测数据上的 PDE 回归和
恢复终态的 PINN 系数读出，**不是神经网络从零训练结果**。原参照仍在
`results/formal/`；完整新训练及 M1 结构/科学断言验收仍是独立未完成项。

[M1 comparison](quick_20260910/M1/comparison.json) passes all 180 numeric values
over 45 parameter rows from 15 independent jobs. Its
[provenance](quick_20260910/M1/provenance.json) preserves two runner identities:
the original nine GIFT/PDE jobs used SHA256 prefix `27AEF71B`, and the later six
PINN jobs used `3D95B708`. This is not a single-source campaign. The first PINN
job's successful numeric commit and its subsequent outer-audit exit 1 are both
retained; no rerun or status rewrite is implied. M1 的 15 个独立任务共 45 行、180
项数值通过原容差；旧九任务与新六任务的源码身份分别保留，不拼成同源从零实验。

`fresh_20260910/` separately records **from-scratch training**, including completed
budgets and explicitly labelled partial runs: per-model budget/input checks,
state comparisons, training history and execution evidence. Its presence does
not mean every model or every fresh-training experiment has passed.

`fresh_20260910/` 则记录实际执行的**从零训练**，分别标明完整预算和部分训练，
保存预算/输入校验、状态比较、训练历史和运行报告，不代表其余模型或全部实验已通过。

The clean, 1% and 10% GIFT low generators each completed 12,000 steps using
16 CPU threads, in 359.47 / 418.15 / 486.19 s respectively; all 21 state tensors
and the model configuration matched the archived counterpart exactly. These
runs use released inputs, not a newly regenerated collection. All three high
branches (seeds 20260820/20260821/20260822) completed 18+2+2 training stages and
passed terminal/provenance/budget checks and qualification. Each nevertheless
has only 5/27 bit-exact state tensors, with history differences from epoch 1.
三条件低频模型的 21 个状态张量及配置分别精确一致；三组高频训练均已完成且资格
校验通过，但每组仅 5/27 状态张量逐位一致，不能改称全轨迹或权重精确复现。

The [fresh-low M1 readout](fresh_20260910/m1_gift_readout/README.md) passed all
36 comparisons over nine parameters (nine values exact); this is only the GIFT
subset. The [new M2 evidence](fresh_20260910/experiments/M2/README.md), using fresh
GIFT weights **and reference FNO2D/FNO3D/U-NO/U-Net weights**, passed 49 rows,
245 numeric values and 98 counts; see its [comparison](fresh_20260910/experiments/M2/comparison.json)
and [input/source provenance](fresh_20260910/experiments/M2/provenance.json).
The [new M3 evidence](fresh_20260910/experiments/M3/README.md), using fresh GIFT
and only published FNO2D/FNO3D baselines, passed all 1,870 rows, 9,350 numeric
values and 3,740 counts in 1,006.35 s; see its [comparison](fresh_20260910/experiments/M3/comparison.json)
and [input/source provenance](fresh_20260910/experiments/M3/provenance.json).
The [new S1 evidence](fresh_20260910/experiments/S1/README.md), using fresh GIFT
only, passed all 180 rows, 180 designated numeric values and 360 counts in
1,236.85 s; see its [comparison](fresh_20260910/experiments/S1/comparison.json)
and [input/source provenance](fresh_20260910/experiments/S1/provenance.json).
All S1 experiment models are freshly trained, but all-project retraining and
complete-project acceptance are not established. Fresh-GIFT S2 completed its
calculation but **failed 6/420 numeric comparisons** (414 passed; all 84 rows and
168 counts matched): mean/sample standard deviation/maximum at leads 2.5/3,
independent holdout, correction disabled, seed 20260821. No outlier was removed
and no tolerance changed. The [fresh-GIFT S3 evidence](fresh_20260910/experiments/S3/README.md)
passed all 33 rows, 132 numeric values and 66 counts; see its
[comparison](fresh_20260910/experiments/S3/comparison.json). S3 PASS does not
cancel S2 FAIL or identify its discrepant raw trajectory.
The earlier `quick_20260910/`
tables remain the reference-weight executions, not these new runs. 新低频模型的
M1 子集、“新 GIFT＋参照基线”的 M2/M3 及仅使用新 GIFT 的 S1 已通过原数值门限；M3 的参照基线仅 FNO2D/FNO3D，这不是所有模型从零训练后
的整套六实验验收，也不证明重新生成数据链已通过。新 GIFT 的 S2 数值有6项超差，
S3的132项数值通过；保留各自真实状态，不以S3通过注销S2失败。最新范围见[复现状态](../docs/REPRODUCIBILITY_STATUS.md)。

[Historical CPU test evidence](environment_20260911/run_02/README.md) records an earlier
existing-environment suite: 143 passed, 25 skipped, three warnings and 24
separately reported passed subtests in 17.72 s. The [earlier evidence](environment_20260911/README.md)
is not overwritten: it retains its 139-pass suite and separate 187-file portable
snapshot, including that snapshot's 15/16 artifact result and missing U-NO file.
Neither test proves a fresh installation or a current full GitHub clone.
CPU 测试与科学验收分开记录，旧文件快照也不覆盖后来新增的测试和证据。

For one available experiment, reproduce the numeric check from the clone root:

```bash
python scripts/compare_results.py M2 validation/quick_20260910/M2/metrics.csv
```

Each `comparison.json` states the unchanged acceptance rule, row counts, checked
values and failures. `provenance.json` records hashes of the actual executed
source and inputs, without copying input fields or external algorithm code.
Machine-local absolute paths and the device UUID have been omitted for portable
publication; hashes of the original local execution records are retained.

这是对随附表格的再次比较，不会启动推理。需要重新计算预测时，请使用
[使用指南](../docs/GETTING_STARTED.md)中的独立实验入口。容差不通过此命令放宽。
当前全项目仍未完成发布验收，详见[复现状态](../docs/REPRODUCIBILITY_STATUS.md)。
