# 当前 W3 的准备、训练、测量与汇总

当前协议是 `single_trajectory_terminal_v1`：固定 50 组参数，每组仅保留 training ID0，观测为 N64、t0–10、dt0.02 的 501 帧；没有独立 validation 轨迹。GIFT 使用 6000+6000 次更新，各阶段固定终点 checkpoint。PINN 使用原有 seed1234、24,000 个训练测量、6,000 个留出测量和 84,000 个物理点，保持原有完整优化预算。参数真值只参与数据 QA 和测量后的 APE，不能作为训练初始化或训练标签。

以下命令从仓库根目录运行。用实际安装路径替换示例变量；两个 Python 可分别指向 PyTorch 环境和具备所需 TensorFlow 原生运行依赖的 PINN 环境。外部算法及运行接口须按 [手工适配说明](EXTERNAL_MANUAL_ADAPTATION.md) 在 GIFT 仓库外准备。GIFT 自身的训练及数据生成无需外部算法实现；PDE-FIND 和 PINN 需要对应外部接口与上游源码。

```powershell
$TorchPython = 'C:\envs\gift\python.exe'
$PinnPython = 'C:\envs\gift-pinn\python.exe'
$Dataset = 'D:\datasets\GIFT-dataset'
$Run = 'D:\runs\w3_current_fresh'
$env:GIFT_EXTERNAL_ROOT = 'D:\upstreams'
$env:GIFT_EXTERNAL_ADAPTER_ROOT = 'D:\gift-external-runtime'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

## 使用 ScienceDB 数据包

先完整核验下载包，再为新的运行目录建立当前协议和输入 gate：

```powershell
& $TorchPython -B -m scripts.verify_data --root $Dataset --full-array-scan
& $TorchPython -B -m scripts.prepare_w3_run --root $Run --dataset-root $Dataset
& $TorchPython -B -m scripts.prepare_w3_run --root $Run --dataset-root $Dataset --execute
```

首条 preparation 命令只显示计划；`--execute` 对全部 50 组 observations、receipt、lock、QA 和 cohort 的身份重新核验，扫描全部观测的有限值、初值参数、ID 和时刻，重算原 FD 数据质量 gate，并用 seed1234 重建全部传感器、时间、训练/留出拆分和 LHS 设计，从每组实际保存场重算 `[u,v,ω]` 采样目标。ScienceDB 包只读。初值/FD 的 float32 机器误差使用已有数据 QA 容限；采样目标的跨 FFT 库检查为 `rtol=5e-6, atol=5e-7`，实际差异写入新记录。

验证全部通过后，新 run 内生成 `PROTOCOL_LOCK.json`、`PREPARATION_SOURCES.json`、`W3_PREPARATION.json`、`data/DATA_COHORT_QA.json` 和 `FULL_INPUT_GATE.json`。每次通过的重验另存 `preparation_checks/`，保留当次 FD/target 浮点诊断；相同输入在已允许容限内的机器舍入差异不会改写首次 preparation 或阻止重验。来源哈希用于记录，不要求普通运行与旧源码逐字节一致。历史数据 receipt 不会修改；gate 不会把不完整的 case 子集认定为完整 50 组。

## 单组执行与恢复

以下以 `parameter_000` 为例。先去掉 `--execute` 可检查计划。GIFT 的命令会从当前 case receipt 选择一轨迹、零 validation、terminal 配置。

```powershell
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method GIFT --action train --device cuda --execute
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method GIFT --action measure --device cpu --execute

& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PDE-FIND --action measure --device cpu --execute
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PDE-FIND-KC --action measure --device cpu --execute

& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR --action train --device gpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR --action measure --device cpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR-KC --action train --device gpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR-KC --action measure --device cpu --execute
```

将 case 依次替换为 `parameter_000`–`parameter_049` 完成全部 250 个方法/实例槽位。每个 case/method 必须使用独立 native journal。中断后只有该 slot 的 native journal 存在时，原训练命令才加 `--resume`；已完成的训练/测量可原命令读取并核验，不重新拟合。OOM、非有限值等已记录科学失败会保留为该固定 slot 的失败，不能删除后重抽 case 或冒充同一 run 的全新初始化。基础设施中断则保留 attempt 日志，并按自己的已保存状态恢复。`--claim-token` 可记录调度器分配标识；单机手工执行不需要历史双机调度锁。

新训练和测量只写 `$Run/training`、`$Run/jobs` 和 `$Run/job_attempts`。不会读取已发表的结果来填充缺失模型或缺失测量。

## 从初值重新生成当前观测

独立生成使用新的目录，例如 `$GeneratedRun = 'D:\runs\w3_current_regenerated'`。保持下载包和旧 run 不变。

```powershell
$GeneratedRun = 'D:\runs\w3_current_regenerated'
& $TorchPython -B -m scripts.generate_w3_parameter_data --root $GeneratedRun --parameter-csv "$Dataset\identification\w3\parameters.csv" --profile single_trajectory_terminal_v1 --device cuda --batch-size 1 --execute
```

该入口直接从固定 ID0 四涡初态重新积分全部 50 个方程，每组一条 training、零 validation；输出当前 schema，明确记录 `new_integration_performed=true`。它不会把新积分伪称为旧数据的逐位无损提取。原 `validation_best_v1` profile 继续可用，未指定 `--profile` 时保持历史 1 train + 20 validation 行为。不能将历史 profile 的结果当作当前主实验。

生成后逐组准备采样，再建立完整 gate：

```powershell
0..49 | ForEach-Object {
    $Case = 'parameter_{0:d3}' -f $_
    & $TorchPython -B -m scripts.run_w3_parameter_job --root $GeneratedRun --case $Case --action prepare-sampling --execute
    if ($LASTEXITCODE -ne 0) { throw "Sampling failed for $Case" }
}
& $TorchPython -B -m scripts.prepare_w3_run --root $GeneratedRun --execute
```

之后沿用前一节的训练/测量命令，将 `--root` 换为 `$GeneratedRun`，移除 `--dataset-root $Dataset`。生成器会保存 run 本地固定参数 CSV、初值、cohort QA；新生成数据与包内历史数据可能有 FFT/设备的 float32 差异，应先检查误差量级和演化积累，再判断是否存在科学异常。重新训练不承诺逐位复现已有权重或每一个历史估计值。

## 当前 50×5 汇总

当前汇总使用专用入口，不能使用旧三种噪声的 15 槽位聚合器：

```powershell
& $TorchPython -B -m scripts.aggregate_w3_current --root $Run --output "$Run\summary_current" --execute
```

它核验 preparation/protocol/cohort 和每个已完成测量的三份数值文件。神经方法还必须具有同 run 的实际模型、训练报告、case 绑定和原生 completion：GIFT 权重及报告与 manifest 的实测哈希一致，PINN 结果与 `COMPLETED.json`、实际 `terminal_state.npz` 三方一致。只声明一个 checkpoint 哈希不能代替这些文件。输出包括 `parameter_cases.csv`、`parameter_results.csv`、`method_metrics.csv`、包含来源及成本的 `numeric_report.json` 和哈希 commit。固定保留 250 slots、750 参数行；科学失败为 `failed`，尚未完成或可恢复的中断为 `pending`，不会用已有发表数据补齐。存在 pending 时报告 `incomplete`；所有 slot 已有终态但含失败时为 `complete_with_failures`。某方法未完成全部 50 个 case 时，该方法正式 mean/median/min/max 留空，成功子集的均值仅放在带 `diagnostic_available_case` 名称的列。

每次汇总使用新的输出目录。汇总保留每槽已记录训练成本、测量耗时和所有 own-run attempt，包括失败和中断；缺少的成本不估算。汇总不训练、不重新积分、不修改历史实验数值。新图可在核验后的 CSV 上另行绘制。
